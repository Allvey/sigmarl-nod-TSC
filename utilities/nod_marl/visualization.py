"""Read-only diagnostics from the opinion cache used to choose an action."""
import torch

from .barrier import aligned_opinions
from .dgppo import opinion_alpha


@torch.no_grad()
def opinion_alpha_lines(td, manager, *, agent_index=0, env_index=0, decision_time=None):
    """Screen text for a pre-action snapshot; never advance NOD or select actions.

    Alpha is the coefficient the training rule would use with the loaded Value,
    not a runtime safety intervention. A missing Value must not look like a
    trustworthy fixed-alpha or safe prediction.
    """
    title = f"Agent {agent_index + 1}: z / alpha (training rule)"
    lines = [title]
    if decision_time is not None:
        lines.append(f"Decision t={decision_time:.2f}s (before displayed step)")
    if manager is None or manager.parameters.safety_control_mode != 'dgppo':
        return lines + ["DGPPO Safety Value unavailable"]
    snapshot = td[env_index:env_index + 1]
    state = manager.state(snapshot)
    n = state['g'].shape[-2]
    if not 0 <= agent_index < n:
        return lines + [f"Agent index must be in 0..{n - 1}"]
    neighbors = state['valid'][0, agent_index, :n].nonzero().flatten().tolist()
    if not neighbors:
        return lines + ["No visible NOD neighbors"]
    available, opinions = aligned_opinions(snapshot, state)
    p = manager.parameters
    opinion_mode = getattr(p, 'dgppo_opinion_alpha', False)
    value_ready = (manager.enabled and manager.model is not None
                   and manager.last_load_info.startswith('loaded'))
    coefficients = details = value = None
    if opinion_mode and value_ready:
        value = manager.model(state)
        coefficients, details = opinion_alpha(
            snapshot, state, value, alpha=p.dgppo_alpha, span=p.dgppo_alpha_span,
            gain=p.dgppo_alpha_gain, deadzone=p.dgppo_opinion_deadzone)
    for j in neighbors:
        known = bool(available[0, agent_index, j])
        z = f"{float(opinions[0, agent_index, j]):+.3f}" if known else "N/A"
        if not opinion_mode:
            alpha, reason = f"{p.dgppo_alpha:.2f}", "fixed mode"
        elif not value_ready:
            alpha, reason = "N/A", "Value unavailable"
        else:
            alpha = f"{float(coefficients[0, agent_index, j]):.2f}"
            if not torch.isfinite(value[0, agent_index, j]):
                alpha, reason = "N/A", "invalid Value"
            elif not known:
                reason = "missing: fixed"
            elif bool(details['applied'][0, agent_index, j]):
                reason = "opinion"
            elif (known and bool(details['safe'][0, agent_index, j])
                  and abs(float(opinions[0, agent_index, j])) <= p.dgppo_opinion_deadzone):
                reason = "deadzone: fixed"
            else:
                reason = "non-safe: fixed"
        suffix = ""
        if getattr(p, "nod_evidence_gate_mode", "fixed") == "ppo":
            gate = snapshot.get(('agents', 'info', 'nod_evidence_gate_context'), default=None)
            weights = snapshot.get(('agents', 'info', 'nod_evidence_weight'), default=None)
            ids = snapshot.get(('agents', 'info', 'nod_neighbor_indices'), default=None)
            if gate is not None and weights is not None and ids is not None:
                slots = (ids[0, agent_index].long() == j).nonzero().flatten()
                if slots.numel():
                    slot = int(slots[0])
                    cached = gate[0, agent_index, slot]
                    if bool(cached[-1]) and known:
                        if getattr(p, "nod_evidence_gate_target", "evidence") == "attention":
                            suffix = (f"  e={float(cached[-4]):+.3f}"
                                      f"  A_RL={float(weights[0, agent_index, slot]):.3f}"
                                      f"  A0={float(cached[-5]):.3f}")
                        else:
                            suffix = (f"  e={float(cached[-4]):+.3f}"
                                      f"  beta={float(weights[0, agent_index, slot]):.3f}"
                                      f"  A={float(cached[-5]):.3f}")
            if not suffix:
                suffix = ("  A_RL=N/A" if getattr(p, "nod_evidence_gate_target", "evidence") == "attention"
                          else "  beta=N/A")
        lines.append(f"A{agent_index + 1} -> A{j + 1}  z={z}  alpha={alpha}  [{reason}]" + suffix)
    return lines
