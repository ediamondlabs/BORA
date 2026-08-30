# Standard libs
import numpy as np

# MINLP
from pyomo.environ import *

# Own Classes
import MANET.Components as net_comps

"""
# MINLP.py

An other approach to the exhaustive search is to formulate an optimization problem, or in this case a, so called, Mixed-Integer Non-Linear Programming Problem.
The implementation is nothing more than the MCMF-problem extended with the capacity depending on the static positions of the jammers and users and the variable positions of the MANETNodes.

However, the current optimizer used (IPOPT) seems unable to solve it. Either, a different solver must be used or the formulation must be improved.
"""


# * MINLP
def findBestPositions(
    nodes: np.ndarray[net_comps.MANETNode],
    jammers: np.ndarray[net_comps.Jammer],
    users: np.ndarray[net_comps.User],
) -> tuple:
    """A MINLP (Mixed Integer Non-Linear Programming) problem to find the best positions for all MANETNodes
    that maximize the throughput. It is basically the MCMF problem extended by a variable capacity that depends
    on the MANETNodes' and Users' position plus the Jammers' positions, in terms of interference.

    Args:
        nodes (np.ndarray[net_comps.MANETNode]): the MANETNode objects for whose positions are to be changed
        jammers (np.ndarray[net_comps.Jammer]): jammer objects
        users (np.ndarray[net_comps.User]): user objects

    Returns:
        tuple: throughput, optimal positions
    """
    # * Constant
    COMPONENTS = np.concatenate((nodes, users))

    # * Initialize the model
    model = ConcreteModel()

    # * Sets
    model.NODES = Set(initialize=[node for node in nodes])
    model.USERS = Set(initialize=[user for user in users])
    model.JAMMERS = Set(initialize=[jammer for jammer in jammers])
    model.COMPONENTS = model.NODES | model.USERS

    model.EDGES = Set(
        initialize=[
            (i, j)
            for i in model.COMPONENTS
            for j in model.COMPONENTS
            if i != j and not (i in model.USERS and j in model.USERS)
        ]
    )

    model.K = Set(initialize=range(len(users)))  # Commodities

    # * Parameters
    model.comp_power_t = Param(
        model.COMPONENTS,
        initialize={comp: comp.signalPower_mW for comp in COMPONENTS},
        within=PositiveReals,
    )
    model.jammer_power_t = Param(
        model.JAMMERS,
        initialize={jammer: jammer.signalPower_mW for jammer in jammers},
        within=PositiveReals,
    )

    model.user_positions = Param(
        model.USERS,
        range(2),
        initialize={(user, dim): user.pos[dim] for user in users for dim in range(2)},
        within=PositiveReals,
    )
    model.jammer_positions = Param(
        model.JAMMERS,
        range(2),
        initialize={
            (jammer, dim): jammer.pos[dim] for jammer in jammers for dim in range(2)
        },
        within=PositiveReals,
    )

    model.N = Param(initialize=1e-7, within=PositiveReals)  # Environment noise
    model.B = Param(initialize=20 * 1e6, within=PositiveIntegers)  # Bandwidth
    model.c = Param(initialize=3 * 1e8, within=PositiveIntegers)  # Speed of light
    model.f = Param(initialize=2.4 * 1e9, within=PositiveIntegers)  # Frequency 2.4 GHz
    model.wavelength = model.c / model.f
    model.lambda_pi = Param(
        initialize=(model.wavelength / (4 * np.pi)) ** 2, within=PositiveReals
    )

    model.DESTINATIONS = Param(
        model.USERS, initialize={user: user.dest for user in users}, within=Any
    )
    model.OFFERED_LOADS = Param(
        model.USERS,
        initialize={user: user.offeredLoad for user in users},
        within=PositiveIntegers,
    )

    # * Variables
    model.flows = Var(model.EDGES, model.K, within=NonNegativeReals)
    model.arcs = Var(model.USERS, within=NonNegativeReals, initialize=0)
    model.node_positions = Var(model.NODES, range(2), within=Reals, bounds=(0, 100))

    model.comp_norms = Var(model.EDGES, within=NonNegativeReals)
    model.squared_comp_norms = Var(model.EDGES, within=NonNegativeReals)

    model.jammer_norms = Var(model.JAMMERS * model.COMPONENTS, within=NonNegativeReals)
    model.squared_jammer_norms = Var(
        model.JAMMERS * model.COMPONENTS, within=NonNegativeReals
    )

    model.powers_received = Var(model.EDGES, within=NonNegativeReals)
    model.interferences = Var(model.JAMMERS * model.COMPONENTS, within=NonNegativeReals)
    model.capacities = Var(model.EDGES, within=NonNegativeReals)

    # * Objective
    def objective_rule(model):
        return sum(model.arcs[user] for user in model.USERS)

    model.objective = Objective(rule=objective_rule, sense=maximize)

    # * Constraints
    # (1) Flow conservation on transit nodes
    def flow_conservation_rule(model, comp, k):
        if comp != users[k] and comp != model.DESTINATIONS[users[k]]:
            incoming = sum(
                model.flows[i, comp, k]
                for i in model.COMPONENTS
                if (i, comp) in model.EDGES
            )
            outgoing = sum(
                model.flows[comp, j, k]
                for j in model.COMPONENTS
                if (comp, j) in model.EDGES
            )
            return incoming == outgoing
        return Constraint.Skip

    model.FlowConservation = Constraint(
        model.COMPONENTS, model.K, rule=flow_conservation_rule
    )

    # (2) Flow conservation at source and sink
    # Flow conservation at source
    def source_rule(model, k):
        source = users[k]

        incoming_source = sum(
            model.flows[i, source, k]
            for i in model.COMPONENTS
            if (i, source) in model.EDGES
        )
        outgoing_source = sum(
            model.flows[source, j, k]
            for j in model.COMPONENTS
            if (source, j) in model.EDGES
        )

        return outgoing_source - incoming_source == model.arcs[source]

    # Flow conservation at sink
    def sink_rule(model, k):
        source = users[k]
        sink = model.DESTINATIONS[source]

        incoming_sink = sum(
            model.flows[i, sink, k]
            for i in model.COMPONENTS
            if (i, sink) in model.EDGES
        )
        outgoing_sink = sum(
            model.flows[sink, j, k]
            for j in model.COMPONENTS
            if (sink, j) in model.EDGES
        )

        return incoming_sink - outgoing_sink == model.arcs[source]

    # Add the constraints to the model
    model.SourceFlow = Constraint(model.K, rule=source_rule)
    model.SinkFlow = Constraint(model.K, rule=sink_rule)

    # (3) Capacity constraints
    def capacity_rule(model, i, j):
        flow = sum(model.flows[i, j, k] for k in model.K)
        return flow <= model.capacities[i, j]

    model.Flow_Cap = Constraint(model.EDGES, rule=capacity_rule)

    # (4) Offered load constraints
    def offered_load_rule(model, user):
        return model.arcs[user] <= model.OFFERED_LOADS[user]

    model.OfferedLoad = Constraint(model.USERS, rule=offered_load_rule)

    # (5.1) Inter component norm
    def comp_norm_rule(
        model, i: net_comps.NetworkComponent, j: net_comps.NetworkComponent
    ):
        pos_i = (
            [model.node_positions[i, 0], model.node_positions[i, 1]]
            if isinstance(i, net_comps.MANETNode)
            else [model.user_positions[i, 0], model.user_positions[i, 1]]
        )

        pos_j = (
            [model.node_positions[j, 0], model.node_positions[j, 1]]
            if isinstance(j, net_comps.MANETNode)
            else [model.user_positions[j, 0], model.user_positions[j, 1]]
        )

        return (
            model.comp_norms[i, j]
            == (pos_i[0] - pos_j[0]) ** 2 + (pos_i[1] - pos_j[1]) ** 2
        )

    model.CompNorm = Constraint(model.EDGES, rule=comp_norm_rule)

    # (6.1) Squared inter component norm
    def square_comp_norm_rule(model, i, j):
        return model.squared_comp_norms[i, j] == model.comp_norms[i, j] ** 2

    model.SquareCompNorm = Constraint(model.EDGES, rule=square_comp_norm_rule)

    # (5.2) Inter jammer norm
    def jammer_norm_rule(
        model, jammer: net_comps.Jammer, comp: net_comps.NetworkComponent
    ):
        pos_jammer = (
            model.jammer_positions[jammer, 0],
            model.jammer_positions[jammer, 1],
        )
        pos_comp = (
            [model.node_positions[comp, 0], model.node_positions[comp, 1]]
            if isinstance(comp, net_comps.MANETNode)
            else [model.user_positions[comp, 0], model.user_positions[comp, 1]]
        )
        return (
            model.jammer_norms[jammer, comp]
            == (pos_jammer[0] - pos_comp[0]) ** 2 + (pos_jammer[1] - pos_comp[1]) ** 2
        )

    model.JammerNorm = Constraint(
        model.JAMMERS, model.COMPONENTS, rule=jammer_norm_rule
    )

    # (6.2) Squared inter jammer norm
    def square_jammer_norm_rule(model, jammer, comp):
        return (
            model.squared_jammer_norms[jammer, comp]
            == model.jammer_norms[jammer, comp] ** 2
        )

    model.SquareJammerNorm = Constraint(
        model.JAMMERS, model.COMPONENTS, rule=square_jammer_norm_rule
    )

    # (7.1) Signal
    def power_received_rule(model, i, j):
        return (
            model.powers_received[i, j]
            == model.comp_power_t[j] * model.lambda_pi / model.squared_comp_norms[i, j]
        )

    model.PowerReceived = Constraint(model.EDGES, rule=power_received_rule)

    # (7.2) Interference
    def interference_rule(model, jammer, comp):
        return (
            model.interferences[jammer, comp]
            == model.jammer_power_t[jammer]
            * model.lambda_pi
            / model.squared_jammer_norms[jammer, comp]
        )

    model.Interference = Constraint(
        model.JAMMERS, model.COMPONENTS, rule=interference_rule
    )

    # (8) Capacity
    def capacity_rule(model, i, j):
        interference_sum = sum(
            model.interferences[jammer, j] for jammer in model.JAMMERS
        )
        return model.capacities[i, j] == model.B * log(
            1 + model.powers_received[i, j] / (model.N + interference_sum)
        ) / log(2)

    model.Capacity = Constraint(model.EDGES, rule=capacity_rule)

    # * Solve with IPOPT
    solver = SolverFactory("ipopt")
    solver.options["tol"] = 1e-6  # Tolerance for convergence
    solver.options["max_iter"] = 10000000  # Maximum iterations
    solver.options["mu_strategy"] = "adaptive"  # Adaptive mu strategy
    solver.options["linear_solver"] = "mumps"  # Use a more robust linear solver
    solver.solve(model, tee=True)

    # * Extract the optimized positions
    optimized_positions = np.array(
        [[model.node_positions[node, dim].value for dim in range(2)] for node in nodes]
    )

    return model.objective(), optimized_positions
