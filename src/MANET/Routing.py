# Easier to use dict for flow_values
from collections import defaultdict

# Measure time

# Math

# Linear optimizers
import xpress as xp
import pulp
from gurobipy import Model, GRB, quicksum

# Graph lib
import networkx as nx
from networkx.algorithms.flow import shortest_augmenting_path, edmonds_karp

# Own classes
from MANET.Components import User


"""
# Routing.py

Routing holds multiple implementations of "Routing-Algorithms" that can be used for the network. Since the thesis focuses on perfect routing, these are not really routing algorithms in the traditional sense but rather "routing" as used in the MANET simulation.

## Implementations of the MCMF Problem

There are three implementations of the Multi-Commodity Min-Cost Flow (MCMF) problem:

### 1. Linear Programming-Based Implementations
These two implementations follow a similar formulation:

- [multi_commodity_max_flow_pulp_1](#MANET.Routing.MCMF.multi_commodity_max_flow_pulp_1)  
  Uses `PuLP` for solving the MCMF problem.

- [multi_commodity_max_flow_xpress](#MANET.Routing.MCMF.multi_commodity_max_flow_xpress)  
  Uses `Xpress` via `PuLP`.  
  Limitation: The XPress solver struggles with handling more than 500 constraints, making it impractical for larger networks.

### 2. Alternative `PuLP` Implementation
- [multi_commodity_max_flow_pulp_2](#MANET.Routing.MCMF.multi_commodity_max_flow_pulp_2)  
  Another formulation using `PuLP`.

### 3. Gurobi Implementation
- [multi_commodity_max_flow_gurobi](#MANET.Routing.MCMF.multi_commodity_max_flow_gurobi)  
  Uses `Gurobi` via `Pyomo`.

All MCMF implementations should be correct, but they tend to be rather slow in practice.

## Faster Max Flow-Based Implementation
Two simpler but faster approaches using only a single commodity max flow algorithm, which both have their downside.
- [sequential_max_flow](#MANET.Routing.MaxFlow.sequential_max_flow):  For each (source, sink, demand) tuple, the max flow algorithm is executed sequentially with flow limited by the demand, and edge capacities are reduced accordingly.
- [artificialartifical_src_sink_max_flow](#MANET.Routing.MaxFlow.artifical_src_sink_max_flow): Adding an artificial source node and an artificial sink node to then only execute a single commodity max flow algorithm.
"""


class MCMF:
    """Holds multiple implementations of the Multi Commodity Max Flow Problem."""

    @staticmethod
    def multi_commodity_max_flow_pulp_1(
        graph: nx.DiGraph, source_sink_demands: list
    ) -> tuple[float, defaultdict]:
        """
        Solve the maximum flow problem using PuLP.

        Args:
            graph (networkx.DiGraph): A directed graph with 'capacity' as edge attribute.
            source_sink_demands (list of tuples): List of (source, sink, offered_load) tuples representing source and sink pairs with an offered load.

        Returns:
            max_max_flow_value (float): The maximum flow achieved given the source and sink pairs.
            flow_values (defaultdict): Flow values per node, where each node holds a dict of destination nodes with flow values.

        Examples:
            >>> G = nx.DiGraph()
            >>> G.add_edge('A', 'B', capacity=10)
            >>> G.add_edge('B', 'C', capacity=5)
            >>> source_sink_demands = [('A', 'C', 5)]
            >>> throughput, flow_values = MCMF.multi_commodity_max_flow_pulp_1(G, source_sink_demands)
            >>> print(throughput)
            5.0
            >>> print(flow_values)
            {'A': {'B': 5}, 'B': {'C': 5}}
        """
        # Extract arcs and capacities from the graph
        capacities = nx.get_edge_attributes(graph, "capacity")
        artif_feedb_arcs = [
            ("arc", sink, source, k)
            for k, (source, sink, offered_load) in enumerate(source_sink_demands)
        ]

        # Initiate the solver
        prob = pulp.LpProblem("Max_Flow", pulp.const.LpMaximize)

        # Create a dictionary for the flow variables
        flow_vars = pulp.LpVariable.dicts(
            "flow",
            [
                (i, j, k)
                for (i, j) in capacities.keys()
                for k in range(len(source_sink_demands))
            ]
            + artif_feedb_arcs,
            0,
        )

        # Objective: Maximize the flow from source to sink
        prob += pulp.lpSum(flow_vars[arc] for arc in artif_feedb_arcs), "Objective"

        # (1) Flow conservation on transit nodes
        for node in graph.nodes:
            for k, (source, sink, offered_load) in enumerate(source_sink_demands):
                if node != source and node != sink:
                    incoming = pulp.lpSum(
                        [
                            flow_vars[(i, node, k)]
                            for i in graph.predecessors(node)
                            if (i, node, k) in flow_vars
                        ]
                    )
                    outgoing = pulp.lpSum(
                        [
                            flow_vars[(node, j, k)]
                            for j in graph.successors(node)
                            if (node, j, k) in flow_vars
                        ]
                    )
                    prob += (
                        incoming == outgoing,
                        f"flow_conservation_{node}_commodity_{k}",
                    )
        # (2) Flow conservation at source and sink nodes
        for k, (source, sink, offered_load) in enumerate(source_sink_demands):
            incoming = pulp.lpSum(
                [
                    flow_vars[(i, sink, k)]
                    for i in graph.predecessors(sink)
                    if (i, sink, k) in flow_vars
                ]
            )
            outgoing = pulp.lpSum(
                [
                    flow_vars[(sink, j, k)]
                    for j in graph.successors(sink)
                    if (sink, j, k) in flow_vars
                ]
            )
            prob += (
                incoming - outgoing == flow_vars[("arc", sink, source, k)],
                f"flow_conservation_at_sink_{sink}_{k}",
            )

            outgoing = pulp.lpSum(
                [
                    flow_vars[(source, j, k)]
                    for j in graph.successors(source)
                    if (source, j, k) in flow_vars
                ]
            )
            incoming = pulp.lpSum(
                [
                    flow_vars[(i, source, k)]
                    for i in graph.predecessors(source)
                    if (i, source, k) in flow_vars
                ]
            )
            prob += (
                outgoing - incoming == flow_vars[("arc", sink, source, k)],
                f"flow_conservation_at_source_{source}_{k}",
            )

        # (3) Capacity constraints
        for i, j in capacities:
            flow_ij = pulp.lpSum(
                [flow_vars[(i, j, k)] for k in range(len(source_sink_demands))]
            )
            prob += flow_ij <= capacities[(i, j)], f"capacity_{i}_{j}"

        # (4) Offered load constraint
        for arc in artif_feedb_arcs:
            prob += (
                flow_vars[arc] <= offered_load,
                f"offered_load_for_{arc[3]}-th_commod",
            )

        solver = pulp.PULP_CBC_CMD(msg=False)

        # Solve the problem
        prob.solve(solver)

        # Retrieve the optimal value of the objective function
        max_flow_value = pulp.value(prob.objective)

        # Initialize the flow values dictionary
        flow_values = defaultdict(lambda: defaultdict(int))

        # Retrieve the flow values for each edge
        for node_u, node_v in graph.edges:
            total_flow = 0
            for k in range(len(source_sink_demands)):
                total_flow += flow_vars[
                    (node_u, node_v, k)
                ].varValue  # Sum over all commodities
            total_flow = int(total_flow)  # Convert to integer
            if total_flow > 0:
                flow_values[node_u][
                    node_v
                ] = total_flow  # Only store values larger than zero

        return max_flow_value, flow_values

    @staticmethod
    def multi_commodity_max_flow_pulp_2(
        graph: nx.DiGraph, source_sink_demands: list
    ) -> tuple[float, defaultdict]:
        """
        Solve the multi-commodity max flow problem.

        Args:
            graph (networkx.DiGraph): A directed graph with 'capacity' as edge attribute.
            source_sink_demands (list of tuples): List of (source, sink) tuples representing source and sink pairs.

        Returns:
            max_flow_value (float): The maximum flow achieved given the source and sink pairs.
            flow_values (defaultdict): Flow values per node, where each node holds a dict of destination nodes with flow values.

        Examples:
            >>> G = nx.DiGraph()
            >>> G.add_edge('A', 'B', capacity=10)
            >>> G.add_edge('B', 'C', capacity=5)
            >>> source_sink_demands = [('A', 'C', 5)]
            >>> throughput, flow_values = MCMF.multi_commodity_max_flow_pulp_2(G, source_sink_demands)
            >>> print(throughput)
            5.0
            >>> print(flow_values)
            {'A': {'B': 5}, 'B': {'C': 5}}
        """
        # Initialize the flow values dictionary, which holds the flows that are pushed
        flow_values = defaultdict(lambda: defaultdict(int))
        # Demands
        demands = [demand for source, sink, demand in source_sink_demands]
        # Filter only connected pairs
        connected_pairs = [
            (source, sink)
            for i, (source, sink, demand) in enumerate(source_sink_demands)
            if nx.has_path(graph, source, sink)
        ]
        k = len(connected_pairs)

        if k == 0:
            return 0, flow_values

        edges = list(graph.edges)
        capacities = nx.get_edge_attributes(graph, "capacity")

        # Initialize the LP problem
        prob = pulp.LpProblem("Max_Flow", pulp.LpMaximize)

        # Create flow variables for each commodity on each edge
        flow_vars = pulp.LpVariable.dicts(
            "flow", [(u, v, i) for u, v in edges for i in range(k)], 0, 1
        )

        # Objective: Maximize the sum of flows arriving at sinks
        prob += (
            pulp.lpSum(
                [
                    flow_vars[(u, sink, i)] * demands[i]
                    for i, (source, sink) in enumerate(connected_pairs)
                    for u in graph.predecessors(sink)
                ]
            ),
            "Total Flow",
        )

        # (1) Link capacity constraint for each edge
        for u, v in edges:
            prob += (
                pulp.lpSum([flow_vars[(u, v, i)] * demands[i] for i in range(k)])
                <= capacities[(u, v)],
                f"Capacity_{u}_{v}",
            )

        # (2) Flow conservation on transit nodes for each commodity
        for node in graph.nodes:
            for i in range(k):
                if node not in [connected_pairs[i][0], connected_pairs[i][1]]:
                    prob += (
                        pulp.lpSum(
                            [flow_vars[(u, node, i)] for u in graph.predecessors(node)]
                        )
                        == pulp.lpSum(
                            [flow_vars[(node, v, i)] for v in graph.successors(node)]
                        )
                    ), f"Conservation_{node}_Commodity_{i}"

        # (3) Flow conservation at the source and sink for each commodity
        for i in range(k):
            source = connected_pairs[i][0]
            sink = connected_pairs[i][1]
            prob += (
                pulp.lpSum(
                    [flow_vars[(source, v, i)] for v in graph.successors(source)]
                )
                == pulp.lpSum(
                    [flow_vars[(u, sink, i)] for u in graph.predecessors(sink)]
                )
            ), f"Source_Sink_Conservation{i}"

        # (4) Flow leaving source and and entering sink 0
        for i in range(k):
            source = connected_pairs[i][0]
            sink = connected_pairs[i][1]
            prob += (
                pulp.lpSum(
                    [flow_vars[(u, source, i)] for u in graph.predecessors(source)]
                )
                == 0
            ), f"Entering_Source{i}"
            prob += (
                pulp.lpSum([flow_vars[(sink, v, i)] for v in graph.successors(sink)])
                == 0
            ), f"Leaving_sink{i}"

        # Solve the problem
        solver = pulp.PULP_CBC_CMD(msg=False)
        prob.solve(solver)

        # Retrieve the optimal value of the objective function
        max_flow_value = pulp.value(prob.objective)

        # Retrieve the flow values for each edge
        for node_u, node_v in graph.edges:
            total_flow = 0
            for i in range(k):
                total_flow += (
                    demands[i] * flow_vars[(node_u, node_v, i)].varValue
                )  # Sum over all commodities
            total_flow = int(total_flow)  # Convert to integer
            if total_flow > 0:
                flow_values[node_u][
                    node_v
                ] = total_flow  # Only store values larger than zero

        return max_flow_value, flow_values

    @staticmethod
    def multi_commodity_max_flow_xpress(
        graph: nx.DiGraph, source_sink_demands: list
    ) -> tuple[float, defaultdict]:
        """
        Solve the maximum flow problem using Xpress.

        Args:
            graph (networkx.DiGraph): A directed graph with 'capacity' as edge attribute.
            source_sink_demands (list of tuples): List of (source, sink, offered_load) tuples representing source and sink pairs with an offered load.

        Returns:
            max_flow_value (float): The maximum flow achieved given the source and sink pairs.
            flow_values (defaultdict): Flow values per node, where each node holds a dict of destination nodes with flow values.

        Examples:
            >>> G = nx.DiGraph()
            >>> G.add_edge('A', 'B', capacity=10)
            >>> G.add_edge('B', 'C', capacity=5)
            >>> source_sink_demands = [('A', 'C', 5)]
            >>> throughput, flow_values = MCMF.multi_commodity_max_flow_xpress(G, source_sink_demands)
            >>> print(throughput)
            5.0
            >>> print(flow_values)
            {'A': {'B': 5}, 'B': {'C': 5}}
        """
        xp.controls.outputlog = 0
        graph = graph.copy()

        # Extract arcs and capacities from the graph
        arcs = list(graph.edges)
        capacities = nx.get_edge_attributes(graph, "capacity")

        # Flow variables
        f = {}
        # Edges
        for i, j in arcs:
            for d in range(len(source_sink_demands)):
                f["edge", i, j, d] = xp.var(name=f"f_{i}_{j}_{d}")
        # Artificial feedback arcs
        for d, (source, sink, offered_load) in enumerate(source_sink_demands):
            f["arc", sink, source, d] = xp.var(name=f"f_arc_{d}")

        p = xp.problem(name="Multic-Commodity Max Flow")
        p.addVariable(f)

        # Flow conservation constraints
        flow = {}
        for k, (source, sink, offered_load) in enumerate(source_sink_demands):
            for node in graph.nodes:
                if node != source and node != sink:
                    incoming_flow = xp.Sum(
                        f["edge", j, node, k] for j in graph.predecessors(node)
                    )
                    outgoing_flow = xp.Sum(
                        f["edge", node, j, k] for j in graph.successors(node)
                    )
                    flow[node, k] = xp.constraint(
                        incoming_flow == outgoing_flow, name=f"cons_at_{node}_{k}"
                    )
                    p.addConstraint(flow[node, k])

        # Flow conservation at source and sink nodes
        for k, (source, sink, offered_load) in enumerate(source_sink_demands):
            incoming = xp.Sum([f["edge", i, sink, k] for i in graph.predecessors(sink)])
            outgoing = xp.Sum([f["edge", sink, j, k] for j in graph.successors(sink)])
            flow["arc", sink, k] = xp.constraint(
                incoming - outgoing == f["arc", sink, source, k],
                name=f"cons_at_sink_{sink}_{k}",
            )
            p.addConstraint(flow["arc", sink, k])

            outgoing = xp.Sum(
                [f["edge", source, j, k] for j in graph.successors(source)]
            )
            incoming = xp.Sum(
                [f["edge", i, source, k] for i in graph.predecessors(source)]
            )
            flow["arc", source, k] = xp.constraint(
                outgoing - incoming == f["arc", sink, source, k],
                name=f"cons_at_source_{source}_{k}",
            )
            p.addConstraint(flow["arc", source, k])

        # Capacity constraints
        capacity = {}
        for i, j in arcs:
            capacity_constraint = (
                xp.Sum(f["edge", i, j, d] for d in range(len(source_sink_demands)))
                <= capacities[(i, j)]
            )
            capacity[i, j] = xp.constraint(
                capacity_constraint, name=f"capacity_{i}_{j}"
            )
            p.addConstraint(capacity[i, j])

        # Demand constraints
        demand = {}
        for d, (source, sink, offered_load) in enumerate(source_sink_demands):
            demand_constraint = f["arc", sink, source, d] <= offered_load
            demand["arc", sink, source, d] = xp.constraint(
                demand_constraint, name=f"offered_load_{d}"
            )
            p.addConstraint(demand["arc", sink, source, d])

        # Objective: Maximize the total flow
        objective = xp.Sum(
            f["arc", sink, source, d]
            for d, (source, sink, offered_load) in enumerate(source_sink_demands)
        )
        p.setObjective(objective, sense=xp.maximize)

        p.solve()

        # Retrieve the optimal value of the objective function
        max_flow_value = p.getObjVal()

        # Initialize the flow values dictionary
        flow_values = defaultdict(lambda: defaultdict(int))

        # Retrieve the flow values for each edge
        for node_i, node_j in arcs:
            total_flow = 0
            for d in range(len(source_sink_demands)):
                # Accumulate the flow value for each commodity on the arc (node_i, node_j)
                total_flow += p.getSolution(f["edge", node_i, node_j, d])
            total_flow = int(total_flow)  # Convert to integer
            if total_flow > 0:
                flow_values[node_i][
                    node_j
                ] = total_flow  # Only store values larger than zero

        return max_flow_value, flow_values

    @staticmethod
    def multi_commodity_max_flow_gurobi(
        graph: nx.DiGraph,
        source_sink_demands: list,
    ) -> tuple[float, defaultdict]:
        """
        Solve the multi-commodity max flow problem using Gurobi.

        Args:
            graph (networkx.DiGraph): A directed graph with 'capacity' as edge attribute.
            source_sink_demands (list of tuples): List of (source, sink) tuples representing source and sink pairs.

        Returns:
            max_flow_value (float): The maximum flow achieved given the source and sink pairs.
            flow_values (defaultdict): The flow values for each edge.

        Examples:
            >>> G = nx.DiGraph()
            >>> G.add_edge('A', 'B', capacity=10)
            >>> G.add_edge('B', 'C', capacity=5)
            >>> multi_commodity_max_flow_gurobi(G, [('A', 'C', 5)])
            (5.0, {('A', 'B'): 5.0, ('B', 'C'): 5.0})
        """
        # Initialize flow values dict, which holds the flows that are pushed
        flow_values = defaultdict(lambda: defaultdict(int))

        # Demands
        demands = [demand for source, sink, demand in source_sink_demands]

        # Filter only connected pairs
        connected_pairs = [
            (source, sink)
            for i, (source, sink, demand) in enumerate(source_sink_demands)
            if nx.has_path(graph, source, sink)
        ]
        k = len(connected_pairs)

        if k == 0:
            return 0, flow_values

        edges = list(graph.edges)
        capacities = nx.get_edge_attributes(graph, "capacity")

        # Initialize the Gurobi model
        model = Model("Max_Flow")
        model.setParam("OutputFlag", False)

        # Create flow variables for each commodity on each edge
        flow_vars = model.addVars(
            [(u, v, i) for u, v in edges for i in range(k)], lb=0, ub=1, name="flow"
        )

        # Objective: Maximize the sum of flows arriving at sinks
        model.setObjective(
            quicksum(
                flow_vars[u, sink, i] * demands[i]
                for i, (source, sink) in enumerate(connected_pairs)
                for u in graph.predecessors(sink)
            ),
            GRB.MAXIMIZE,
        )

        # (1) Link capacity constraint for each edge
        for u, v in edges:
            model.addConstr(
                quicksum(flow_vars[u, v, i] * demands[i] for i in range(k))
                <= capacities[(u, v)],
                f"Capacity_{u}_{v}",
            )

        # (2) Flow conservation on transit nodes for each commodity
        for node in graph.nodes:
            for i in range(k):
                if node not in [connected_pairs[i][0], connected_pairs[i][1]]:
                    model.addConstr(
                        quicksum(
                            flow_vars[u, node, i] for u in graph.predecessors(node)
                        )
                        == quicksum(
                            flow_vars[node, v, i] for v in graph.successors(node)
                        ),
                        f"Conservation_{node}_Commodity_{i}",
                    )

        # (3) Flow conservation at the source and sink for each commodity
        for i in range(k):
            source = connected_pairs[i][0]
            sink = connected_pairs[i][1]
            model.addConstr(
                quicksum(flow_vars[source, v, i] for v in graph.successors(source))
                == quicksum(flow_vars[u, sink, i] for u in graph.predecessors(sink)),
                f"Source_Sink_Conservation{i}",
            )

        # (4) Flow leaving source and entering sink 0
        for i in range(k):
            source = connected_pairs[i][0]
            sink = connected_pairs[i][1]
            model.addConstr(
                quicksum(flow_vars[u, source, i] for u in graph.predecessors(source))
                == 0,
                f"Entering_Source{i}",
            )
            model.addConstr(
                quicksum(flow_vars[sink, v, i] for v in graph.successors(sink)) == 0,
                f"Leaving_sink{i}",
            )

        # Solve the problem
        model.optimize()

        # Retrieve the optimal value of the objective function
        max_flow_value = model.objVal

        # Collect and sum flow values per edge
        for node_u, node_v in edges:
            total_flow = 0
            for i in range(k):
                total_flow += (
                    flow_vars[node_u, node_v, i].X * demands[i]
                )  # Scale by demand
            total_flow = int(total_flow)  # Convert to integer
            if total_flow > 0:
                flow_values[node_u][
                    node_v
                ] = total_flow  # Only store values larger than zero

        return max_flow_value, flow_values


class MaxFlow:
    """Holds more simpler max flow solutions for perfect routing"""

    @staticmethod
    def sequential_max_flow(
        graph: nx.DiGraph, source_sink_demands: list[tuple[User, User, int]]
    ) -> tuple[float, defaultdict]:
        """
        A sequential max flow routing algorithm, assuming nearly perfect routing.
        For each (source, sink, demand) tuple, the max flow algorithm is executed sequentially
        with flow limited by the demand, and edge capacities are reduced accordingly.

        Args:
            graph (nx.DiGraph): The network graph.
            source_sink_demands (list[tuple[User, User, int]]): List of (source, sink, demand) tuples.

        Returns:
            throughput (float): Total data successfully sent.
            flow_values (defaultdict): Flow values per node, where each node holds a dict of destination nodes with flow values.

        Examples:
            >>> G = nx.DiGraph()
            >>> G.add_edge('A', 'B', capacity=10)
            >>> G.add_edge('B', 'C', capacity=5)
            >>> source_sink_demands = [('A', 'C', 5)]
            >>> throughput, flow_values = MaxFlow.sequential_max_flow(G, source_sink_demands)
            >>> print(throughput)
            5.0
            >>> print(flow_values)
            {'A': {'B': 5}, 'B': {'C': 5}}
        """
        throughput = 0
        flow_values = defaultdict(lambda: defaultdict(int))  # Avoid redundant checks

        for source, sink, demand in source_sink_demands:
            if nx.has_path(graph, source, sink):
                # Determine flow_func
                edges = graph.edges
                nodes = graph.nodes
                flow_func = (
                    shortest_augmenting_path
                    if len(edges) > len(nodes)
                    else edmonds_karp
                )
                # Get flow_value
                max_flow_value, flow_dict = nx.maximum_flow(
                    graph,
                    source,
                    sink,
                    capacity="capacity",
                    flow_func=flow_func,
                )

                # Ensure data_to_send does not exceed demand
                data_to_send = min(max_flow_value, demand)

                # Compute the ratio of actual sent data to available max flow
                available_to_max_ratio = (
                    data_to_send / max_flow_value if max_flow_value > 0 else 0
                )

                if data_to_send > 0:
                    for node_u, neighbors in flow_dict.items():
                        for node_v, raw_flow in neighbors.items():
                            # Scale flow
                            flow_amount = int(raw_flow * available_to_max_ratio)

                            if flow_amount > 0:
                                new_capacity = (
                                    graph[node_u][node_v]["capacity"] - flow_amount
                                )
                                if new_capacity <= 0:
                                    graph.remove_edge(node_u, node_v)
                                else:
                                    graph[node_u][node_v]["capacity"] = new_capacity

                                # Store flow values in expected format
                                flow_values[node_u][node_v] += flow_amount

                    # Update total throughput
                    throughput += data_to_send

        return throughput, flow_values

    @staticmethod
    def artifical_src_sink_max_flow(
        graph: nx.DiGraph, source_sink_demands: list[tuple[User, User, int]]
    ) -> tuple[float, defaultdict]:
        """A even speedier implementation using the max flow algorithm instead of any MCMF
        solution. However, this assumes that the set of source and the set of sinks are disjoint.
        Otherwise, the result would be faulty.

        How it works:

        An artificial source node and an artificial sink node are added. The artificial source connects
        to all source nodes and so has an edge from the artificial source node to each source node.
        The edge's capacity is imposed with the demand value, that the source node wants to send. This
        enforces that what each source node sends is limited by the demand as it should. Similar,
        the sink nodes each have an edge to the artificial sink also limited witht the capacity equal
        to the indivdiual demand.

        Problem:

        I am not 100% sure, but I believe that the current implementation, while correctly limiting with
        the demand, doesn't ensure that the flow leaving source i is the flow entering sink i. Commodities
        might mix their flows.

            Example:

            src_1 -> A,
            A -> sink_1
            src_2 -> A

            now, what if  src_1 edge capacity to A is smaller than the demand,
            hence src_2 could add of its demand to the receiving flow at sink_1,
            even to src_2 should send to sink_2

        Args:
            graph (nx.DiGraph): Directed graph with 'capacity' edge attribute
            source_sink_demands (list): List of (source, sink, demand) tuples

        Returns:
            throughput (float): Total data successfully sent.
            flow_values (defaultdict): Flow values per node, where each node holds a dict of destination nodes with flow values.

        Examples:
            >>> G = nx.DiGraph()
            >>> G.add_edge('A', 'B', capacity=10)
            >>> G.add_edge('B', 'C', capacity=5)
            >>> source_sink_demands = [('A', 'C', 5)]
            >>> throughput, flow_values = MaxFlow.art_src_sink_max_flow(G, source_sink_demands)
            >>> print(throughput)
            5.0
            >>> print(flow_values)
            {'A': {'B': 5}, 'B': {'C': 5}}
        """
        # Assert sources and sinks are disjoint
        sources = set(src for src, _, _ in source_sink_demands)
        sinks = set(snk for _, snk, _ in source_sink_demands)
        assert sources.isdisjoint(sinks), "Sources and sinks gotta be disjoint!"
        # Set up artifical src and sink and connect them to the actual sources
        # and sinks initialized with each commodities demand as the edge capacity.
        art_src = "art_src"
        art_sink = "art_sink"

        graph.add_nodes_from([art_src, art_sink])

        for src, sink, demand in source_sink_demands:
            # Artif src connects to each src with its demand
            graph.add_edge(art_src, src, capacity=demand)
            # Each sink connects to the artif sink with its demand.
            graph.add_edge(sink, art_sink, capacity=demand)

        # Determine flow_func
        edges = graph.edges
        nodes = graph.nodes
        flow_func = (
            shortest_augmenting_path if len(edges) > len(nodes) else edmonds_karp
        )

        # Get flow_value
        max_flow_value, flow_dict = nx.maximum_flow(
            graph,
            art_src,
            art_sink,
            capacity="capacity",
            flow_func=flow_func,
        )

        return max_flow_value, flow_dict


def graph_to_mcf_string(graph: nx.DiGraph, source_sink_demands: list) -> str:
    """
    Convert NetworkX DiGraph and demands to MCF solver string format.

    Args:
        graph (nx.DiGraph): Directed graph with 'capacity' edge attribute
        source_sink_demands (list): List of (source, sink, demand) tuples

    Returns:
        str: Formatted string for MCF solver
    """
    # Map nodes to indices
    nodes_to_idx = {node: idx for idx, node in enumerate(graph.nodes)}
    # Build node section
    num_nodes = len(graph.nodes)
    node_lines = [f"{num_nodes}"]
    for idx, node in enumerate(graph.nodes):
        # Using dummy coordinates (100, 100) as they're not used
        node_lines.append(f"{idx} 100 100")

    # Build edge section
    edges = list(graph.edges(data=True))
    num_edges = len(edges)
    edge_lines = [f"{num_edges}"]
    for edge_id, (src, dst, data) in enumerate(edges):
        capacity = data.get("capacity", 0)
        # Using delay=1.0 as default cost if not specified
        delay = 1
        src_idx = nodes_to_idx[src]
        dst_idx = nodes_to_idx[dst]
        edge_lines.append(f"{edge_id} {src_idx} {dst_idx} {capacity} {delay}")

    # Build demand section
    num_demands = len(source_sink_demands)
    demand_lines = [f"{num_demands}"]
    for demand_id, (src, sink, demand) in enumerate(source_sink_demands):
        src_idx = nodes_to_idx[src]
        sink_idx = nodes_to_idx[sink]
        demand_lines.append(f"{demand_id} {src_idx} {sink_idx} {demand}")

    # Combine all sections with newlines
    return "\n".join(node_lines + edge_lines + demand_lines)


def transform_flow_dict(flow_dict: dict, idx_to_nodes: dict) -> defaultdict:
    """Transform a flow dict with different indices to the NetworkComponent object indices

    Args:
        flow_dict (dict): Flow dict with different indices
        idx_to_nodes (dict): Dict of how to remap the indices to the Net

    Returns:
        defaultdict: Transformed flow_dict with the graph nodes being a NetworkComponent
    """
    # Using defaultdict for efficiency
    transformed_dict = defaultdict(lambda: defaultdict(float))
    for source, targets in flow_dict.items():
        for target, flow in targets.items():
            transformed_dict[idx_to_nodes[source]][idx_to_nodes[target]] = flow
    return transformed_dict
