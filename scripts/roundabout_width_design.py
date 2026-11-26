"""Analytic parallel-lane geometry, guided by the original aerial photo.

The editable design is trace.json. Markings, pavement and executable lanelets
all come from these same spines, rather than independent image detections.
"""
from functools import lru_cache
import json
import math
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAME = "roundabout_realistic"
TRACE = json.loads((ROOT/"assets/maps"/(NAME+".trace.json")).read_text())
REFERENCE = ROOT/TRACE["reference_image"]
WIDTH = TRACE["lane_width"]
PIXEL_SCALE = TRACE["simulation_units_per_pixel"]
MARGIN = TRACE["world_margin"]
SPACING = .035
RASTER_SCALE = 4
SPINE_AXES = np.asarray(TRACE["ring_spine_axes_pixels"])*PIXEL_SCALE
RADII = np.array([SPINE_AXES.mean()-WIDTH/2, SPINE_AXES.mean()+WIDTH/2])
ISLAND_RADIUS = np.mean(TRACE["island_axes_pixels"])*PIXEL_SCALE


def world(points):
    return (np.asarray(points, float)*[PIXEL_SCALE,-PIXEL_SCALE]
            +[MARGIN,MARGIN+TRACE["image_size"][1]*PIXEL_SCALE])


def pixels(points):
    return ((np.asarray(points)-[MARGIN,MARGIN+TRACE["image_size"][1]*PIXEL_SCALE])
            /[PIXEL_SCALE,-PIXEL_SCALE])


def normalized(v):
    return np.asarray(v)/np.linalg.norm(v,axis=-1,keepdims=True)


def frames(points):
    t = normalized(np.gradient(points,axis=0))
    return t,np.column_stack((-t[:,1],t[:,0]))


def resample(points):
    arc = np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    s = np.linspace(0.,arc[-1],max(3,int(math.ceil(arc[-1]/SPACING))+1))
    return np.column_stack([np.interp(s,arc,points[:,d]) for d in range(2)])


def ring_frame(angles):
    a = np.atleast_1d(angles)
    p = world(TRACE["ring_center"])+np.column_stack((np.cos(a),np.sin(a)))*SPINE_AXES
    t = normalized(np.column_stack((-np.sin(a),np.cos(a)))*SPINE_AXES)
    return p,t,np.column_stack((-t[:,1],t[:,0]))


def ring_point(lane,angle):
    p,_,n = ring_frame(angle)
    return (p+(WIDTH/2 if lane == 0 else -WIDTH/2)*n)[0]


def bezier_frame(start,u,end,v,handles):
    first,last = np.asarray(handles)*PIXEL_SCALE
    c = np.array([start,start+first*u,end-last*v,end])
    t = np.linspace(0.,1.,1001)[:,None]
    p = (1-t)**3*c[0]+3*(1-t)**2*t*c[1]+3*(1-t)*t*t*c[2]+t**3*c[3]
    dp = 3*(1-t)**2*(c[1]-c[0])+6*(1-t)*t*(c[2]-c[1])+3*t*t*(c[3]-c[2])
    tangent = normalized(dp)
    return p,tangent,np.column_stack((-tangent[:,1],tangent[:,0]))


@lru_cache(maxsize=1)
def carriageways():
    result = {}
    for i,arm in enumerate(TRACE["approaches"]):
        for kind in ("entry","exit"):
            p,t,_ = ring_frame(math.radians(arm[kind+"_angle_deg"]))
            portal = world(arm[kind+"_portal"])
            if kind == "entry":
                args = portal,normalized(arm["entry_inward_direction"]),p[0],t[0]
            else:
                args = p[0],t[0],portal,normalized(arm["exit_outward_direction"])
            result[i,kind] = bezier_frame(*args,arm[kind+"_handles_pixels"])
    return result


def footprint_corners(points):
    t,n = frames(points)
    half_length,half_width = np.asarray(TRACE["vehicle_size"])/2
    return np.stack([points+s*half_length*t+r*half_width*n
                     for s in (-1,1) for r in (-1,1)])


def clearance_values(points,distance):
    p = np.rint(pixels(points)*RASTER_SCALE).astype(int)
    inside = ((p[...,0]>=0)&(p[...,0]<distance.shape[1])
              &(p[...,1]>=0)&(p[...,1]<distance.shape[0]))
    values = np.zeros(p.shape[:-1])
    values[inside] = distance[p[...,1][inside],p[...,0][inside]]/RASTER_SCALE
    return values


def build_geometry():
    ways,desc,entries,exits,arcs = {},{},{},{},{}

    def add(points,**description):
        key = str(len(ways)+1)
        ways[key],desc[key] = np.asarray(points),description
        return key

    events = sorted([(math.radians(a[k+"_angle_deg"])%(2*np.pi),i,k)
                     for i,a in enumerate(TRACE["approaches"]) for k in ("entry","exit")])
    event_index = {(arm,kind):k for k,(_,arm,kind) in enumerate(events)}
    for lane in range(2):
        offset = WIDTH/2 if lane == 0 else -WIDTH/2
        for k,(start,_,_) in enumerate(events):
            stop = events[(k+1)%len(events)][0]
            if stop <= start:
                stop += 2*np.pi
            p,_,n = ring_frame(np.linspace(start,stop,501))
            points = resample(p+offset*n)
            points[-1] = ring_point(lane,events[(k+1)%len(events)][0])
            arcs[lane,k] = add(points,kind="ring",lane=lane,event=k)
        for arm_id,arm in enumerate(TRACE["approaches"]):
            for kind,mapping in (("entry",entries),("exit",exits)):
                p,_,n = carriageways()[arm_id,kind]
                points = resample(p+offset*n)
                points[-1 if kind == "entry" else 0] = ring_point(lane,math.radians(arm[kind+"_angle_deg"]))
                mapping[lane,arm_id] = add(points,kind=kind,lane=lane,arm=arm_id,
                                           # Use the approach up to six samples before
                                           # the merge, rather than crowding 20 cars
                                           # into its first third. Never spawn on ring.
                                           spawn_end_index=len(points)-6)
    routes,info = [],[]
    for lane in range(2):
        for origin in range(4):
            for destination in range(4):
                if origin == destination:
                    continue
                k,stop = event_index[origin,"entry"],event_index[destination,"exit"]
                ring_ids = []
                while k != stop:
                    ring_ids.append(arcs[lane,k]); k = (k+1)%len(events)
                entry = entries[lane,origin]
                routes.append([entry]+ring_ids+[exits[lane,destination]])
                info.append(dict(origin=origin,destination=destination,lane=lane,
                                 spawn_start_index=4,spawn_end_index=desc[entry]["spawn_end_index"]))
    neighbors = {key:[other for other,q in ways.items()
                      if np.linalg.norm(p[:,None]-q[None,:],axis=-1).min()<.8]
                 for key,p in ways.items()}
    dims = np.array(TRACE["image_size"])*PIXEL_SCALE+2*MARGIN
    return ways,desc,routes,info,neighbors,world(TRACE["ring_center"]),dims


@lru_cache(maxsize=1)
def design_layers():
    """Union of constant-width pavement ribbons, rasterized at quarter pixels."""
    w,h = TRACE["image_size"]
    shape = h*RASTER_SCALE,w*RASTER_SCALE

    def blank():
        return np.zeros(shape,np.uint8)

    def fill(mask,points,value=255):
        p = np.rint(pixels(points)*RASTER_SCALE).astype(np.int32)
        if np.any(p<2) or np.any(p>=np.array([shape[1],shape[0]])-2):
            raise ValueError("Road design exceeds frame: move a portal or increase the frame")
        cv2.fillPoly(mask,[p],value)

    def stroke(mask,points,width=.012,closed=False):
        p = np.rint(pixels(points)*RASTER_SCALE).astype(np.int32)
        cv2.polylines(mask,[p],closed,255,max(1,round(width/PIXEL_SCALE*RASTER_SCALE)))

    road,apron,island,crossing,white,yellow,curb,island_curb = [blank() for _ in range(8)]
    rp,_,rn = ring_frame(np.linspace(0.,2*np.pi,2401))
    fill(road,rp-WIDTH*rn); fill(road,rp+WIDTH*rn,0)
    fill(apron,rp+WIDTH*rn)
    for (_,kind),(p,t,n) in carriageways().items():
        # Road extends beyond a route terminal so an entire vehicle fits at
        # the last centre point, rather than hanging over a rectangular cap.
        if kind == "entry":
            pavement = np.vstack((p[0]-.12*t[0],p))
            normals = np.vstack((n[0],n))
        else:
            pavement = np.vstack((p,p[-1]+.12*t[-1]))
            normals = np.vstack((n,n[-1]))
        fill(road,np.vstack((pavement+WIDTH*normals,(pavement-WIDTH*normals)[::-1])))
    # Round concave curb junctions without removing any lane ribbon.
    radius = round(.04/PIXEL_SCALE*RASTER_SCALE)
    road = cv2.morphologyEx(road,cv2.MORPH_CLOSE,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1)))
    apron[road>0] = 0
    for arm_id in range(4):
        ep = carriageways()[arm_id,"entry"][0][550:]
        xp = carriageways()[arm_id,"exit"][0][:451]
        fill(island,np.vstack((ep,xp)))
    island[(road>0)|(apron>0)] = 0
    a = np.linspace(0.,2*np.pi,721)
    green = world(TRACE["ring_center"])+np.column_stack((np.cos(a),np.sin(a)))*np.asarray(TRACE["island_axes_pixels"])*PIXEL_SCALE
    fill(island,green)
    stroke(island_curb,rp+WIDTH*rn,.012,True)
    stroke(island_curb,green,.012,True)
    contours,_ = cv2.findContours(road,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(curb,contours,-1,255,max(1,round(.012/PIXEL_SCALE*RASTER_SCALE)))

    def dashed(points):
        s = np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
        for start in np.arange(.02,s[-1],.14):
            sel = (s>=start)&(s<=start+.075)
            if sel.sum()>=2:
                stroke(white,points[sel],.009)

    dashed(rp)
    for (arm_id,kind),(p,t,n) in carriageways().items():
        # End approach paint before the merging area. Drawing the complete
        # entry/exit spines over the ring produces misleading narrow lenses.
        relative = (p-world(TRACE["ring_center"]))/SPINE_AXES
        outside_ring = np.linalg.norm(relative,axis=1) > 1+(WIDTH+.025)/SPINE_AXES.min()
        dashed(p[outside_ring])
        anchor = world(TRACE["approaches"][arm_id][kind+"_crossing_pixel"])
        j = int(np.linalg.norm(p-anchor,axis=1).argmin())
        c = p[j]
        fill(crossing,np.array([c+s*.045*t[j]+r*WIDTH*n[j]
                               for s,r in ((-1,-1),(1,-1),(1,1),(-1,1))]))
        j = 310 if kind == "entry" else 720
        for offset in (WIDTH/2,-WIDTH/2):
            c,u,v = p[j]+offset*n[j],t[j],n[j]
            arrow = np.array([[-.065,-.009],[.015,-.009],[.015,-.029],
                              [.065,0.],[.015,.029],[.015,.009],[-.065,.009]])
            fill(white,c+arrow[:,:1]*u+arrow[:,1:]*v)
    crossing[road==0] = 0
    white[road==0] = 0
    return road,[
        ("road",road,[.64,.64,.64],20.),
        ("apron",apron,[.96,.91,.84],100.),
        ("island",island,[.71,.84,.60],20.),
        ("crossing",crossing,[.88,.55,.55],20.),
        ("white_marking",white,[.98,.98,.98],2.),
        ("lane_marking",yellow,[1.,.88,.12],2.),
        ("curb",curb,[.03,.82,.84],2.),
        ("island_curb",island_curb,[.88,.12,.10],2.)]


def validate_design(ways, routes):
    """Reject unbuildable curves before overwriting any executable assets."""
    if WIDTH < 2*TRACE["vehicle_size"][1]:
        raise ValueError("Lane width must leave lateral room beyond the vehicle width")
    road, _ = design_layers()
    distance = cv2.distanceTransform(road, cv2.DIST_L2, 5)
    max_curvature, min_clearance = 0., float("inf")
    for route in routes:
        p = np.vstack([ways[route[0]]]+[ways[k][1:] for k in route[1:]])
        v = np.diff(p, axis=0)
        length = np.linalg.norm(v, axis=1)
        if not np.isfinite(p).all() or length.min() < .015:
            raise ValueError("Invalid or repeated route samples")
        yaw = np.unwrap(np.arctan2(v[:, 1], v[:, 0]))
        curvature = np.abs(np.diff(yaw))/((length[1:]+length[:-1])/2)
        max_curvature = max(max_curvature, float(curvature.max()))
        # Use the simulator's forward-segment yaw, including route junctions.
        t = normalized(np.vstack((v, v[-1])))
        n = np.column_stack((-t[:, 1], t[:, 0]))
        half_length, half_width = np.asarray(TRACE["vehicle_size"])/2
        body = np.stack([p+s*half_length*t+r*half_width*n
                         for s in np.linspace(-1, 1, 5)
                         for r in np.linspace(-1, 1, 3)])
        min_clearance = min(min_clearance, float(clearance_values(body, distance).min()*PIXEL_SCALE))
    if max_curvature > 2.6 or min_clearance < .025:
        raise ValueError(f"Insufficient steering/road clearance: curvature={max_curvature}, clearance={min_clearance}")
    for p, t, n in list(carriageways().values())+[ring_frame(np.linspace(0, 2*np.pi, 1001))]:
        curvature = np.linalg.norm(np.gradient(t, axis=0), axis=1)/np.linalg.norm(np.gradient(p, axis=0), axis=1)
        if np.max(curvature)*WIDTH >= .5:
            raise ValueError("Parallel road boundary folds or has insufficient radius")
        left, right = p+WIDTH/2*n, p-WIDTH/2*n
        nearest = np.linalg.norm(left[::4, None]-right[None, ::4], axis=-1).min()
        if nearest < WIDTH-.002:
            raise ValueError("Adjacent lane corridors approach each other too closely")
    return dict(lane_width_m=WIDTH, vehicle_size_m=TRACE["vehicle_size"],
                nominal_side_margin_m=(WIDTH-TRACE["vehicle_size"][1])/2,
                minimum_sampled_body_to_road_clearance_m=round(min_clearance, 6),
                maximum_sampled_route_curvature_per_m=round(max_curvature, 6),
                scope="Offline nominal geometry only; not a closed-loop collision guarantee")
