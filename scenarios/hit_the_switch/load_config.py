# config_loader_condensed.py
from collections.abc import Mapping
from vmas.simulator.utils import ScenarioUtils
from scenarios.exploration.grids.language_grid import DEFEND_TIGHT, DEFEND_WIDE

# --------------------------------------------------------------------------- #
# 1. PARAMS table
# --------------------------------------------------------------------------- #
PARAMS = [
    
    ("x_semidim", 1.0), ("y_semidim", 1.0),
    ("n_passages", 1), ("shared_reward", False),
    ("n_agents", 5), ("agent_radius", 0.03333),
    ("agent_spacing", 0.1),("passage_width", 0.2),
    ("passage_length", 0.103),("switch_length", 0.2),
    ("switch_width", 0.2), ("break_all_wall", False),
    
    # --- Rewards ------------------------------------------------------------
    ("shaping_factor", 100),
    
    # --- Dynamics -----------------------------------------------------------
    ("use_velocity_controller", True), ("use_kinematic_model", False),
    ("agent_weight", 1.0), ("agent_v_range", 1.0), ("agent_a_range", 1.0),
    ("min_collision_distance", 0.1), ("linear_friction", 0.1),
    
    # --- Communication / GNN -----------------------------------------------
    ("use_gnn", False), ("comm_dim", 1),
    ("comms_radius", 0.35),
    
    # --- Language -----------------------------------------------------------
    ("embedding_size", 1024), ("use_embedding_ratio", 1.0),
    ("event_dim", 3), ("state_dim", 4), ("use_rnn", False),
    
    # --- Paths -----------------------------------------------------------
    ("data_json_path", ""), ("decoder_model_path", ""), ("sequence_model_path", ""),
]

# Expand every entry to canonical (dest, key, default) form
PARAMS = [
    (dest, key, default)
    for entry in PARAMS
    for dest, key, default in [(
        # ────────────────────────────────────────────────
        # 3-tuples stay as-is
        entry if isinstance(entry, tuple) and len(entry) == 3 else
        # 2-tuples → (name, name, default)
        (entry[0], entry[0], entry[1]) if isinstance(entry, tuple) else
        # bare-string → (name, name, None)
        (entry, entry, None)
    )]
]

# --------------------------------------------------------------------------- #
# 2.  Generic loader
# --------------------------------------------------------------------------- #
def load_scenario_config(source, env):
    """
    Parameters
    ----------
    source : dict-like (kwargs) | object with attributes
    env    : your VMAS scenario environment
    """
    is_mapping = isinstance(source, Mapping)

    for dest_attr, key, default in PARAMS:
        if is_mapping:
            value = source.get(key, default)
        else:
            value = getattr(source, key, default)
        setattr(env, dest_attr, value)

    # --- derived attributes -------------------------------------------------
    env.agent_f_range = env.agent_a_range + env.linear_friction
    env.agent_u_range = (
        env.agent_v_range if env.use_velocity_controller else env.agent_f_range
    )
    env.viewer_zoom = 1
