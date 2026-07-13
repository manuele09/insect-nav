"""
GeNN neuron, synapse, and custom-update model definitions.

Requires: pip install insect_nav[genn]
"""

try:
    from pygenn import (
        VarAccessMode,
        create_current_source_model,
        create_custom_update_model,
        create_neuron_model,
        create_weight_update_model,
    )
except ImportError as _err:
    raise ImportError(
        "genn_models requires pygenn. Install with: pip install insect_nav[genn]"
    ) from _err


cs_model = create_current_source_model(
    "cs_model",
    vars=[("magnitude", "scalar")],
    injection_code="injectCurrent(magnitude);",
)

if_model = create_neuron_model(
    "IF",
    params=["TauM", "Rmembrane", "Vthresh", "Vreset"],
    vars=[("V", "scalar")],
    sim_code="V += (dt/TauM)*Rmembrane*Isyn;",
    threshold_condition_code="V >= Vthresh",
    reset_code="V = Vreset;",
)

anti_hebbian = create_weight_update_model(
    "anti_hebbian",
    params=["mod", "halve_g"],
    # "halved" is a per-synapse guard: at most one depression event per
    # presentation may halve g (prevents 0.5**N compounding when a KC and/or
    # the MBON spike more than once within PRESENT_TIME_MS -- a coincident
    # KC-MBON spike is otherwise detected once per spike, not once per
    # presentation). Reset to 0 at the start of every presentation by the
    # "reset_kc_mbon_halved" custom update (see NeuralNetwork._build_network).
    # Irrelevant when halve_g<=0 (classic g=0 is idempotent, no guard needed).
    vars=[("g", "scalar"), ("halved", "scalar")],
    pre_spike_syn_code="""
        addToPost(g);
        if (mod > 0)
        {
            const scalar dt = t - st_post;
            if (dt > 0 && dt < 100000) {
                if (halve_g > 0) {
                    if (halved < 1.0) { g *= 0.5; halved = 1.0; }
                } else {
                    g = 0;
                }
            }
        }
    """,
    post_spike_syn_code="""
        if (mod > 0)
        {
            const scalar dt = t - st_pre;
            if (dt > 0 && dt < 100000) {
                if (halve_g > 0) {
                    if (halved < 1.0) { g *= 0.5; halved = 1.0; }
                } else {
                    g = 0;
                }
            }
        }
    """,
)

pwl_model = create_neuron_model(
    "PWL",
    params=["R", "v0", "eps", "tau_i", "Vknee", "m0", "m1", "m2", "i0"],
    vars=[("V", "scalar"), ("U", "scalar"), ("Vpre", "scalar")],
    sim_code="""
    const scalar imax = m0 * Vknee;
    scalar y;
    if(V > -Vknee && V < Vknee){
        y = m0 * V;
    }else if(V <= -Vknee)
        y = -imax + m1 *(V + Vknee);
    else if(V >= Vknee)
        y = imax + m2 *(V - Vknee);

    Vpre = V;
    V+= dt*(1/(eps*tau_i))*(y - U + i0 + Isyn);
    U+= dt*(1/tau_i)*(V - R * U +v0);
    """,
    threshold_condition_code="Vpre < -Vknee && V > -Vknee",
    reset_code="",
)

reset_model_lif = create_custom_update_model(
    "reset_lif",
    var_refs=[
        ("V", "scalar", VarAccessMode.READ_WRITE),
        ("RefracTime", "scalar", VarAccessMode.READ_WRITE),
    ],
    update_code="""
    V = -60.0f;
    RefracTime = 0.0f;
    """,
)

reset_model_if = create_custom_update_model(
    "reset_if",
    var_refs=[("V", "scalar", VarAccessMode.READ_WRITE)],
    update_code="V = 0.0f;",
)

reset_model_syn = create_custom_update_model(
    "reset_syn",
    var_refs=[("Isyn", "scalar", VarAccessMode.READ_WRITE)],
    update_code="Isyn = 0.0f;",
)

reset_model_wu_var = create_custom_update_model(
    "reset_wu_var",
    var_refs=[("var", "scalar", VarAccessMode.READ_WRITE)],
    update_code="var = 0.0f;",
)
