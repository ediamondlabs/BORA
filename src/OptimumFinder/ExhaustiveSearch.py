# Standard libs
import math
import numpy as np

# Exhaustive search
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, Point, LineString

from itertools import permutations, combinations, product

# For type annotation
from typing import Union, Generator

# Progress bar
from tqdm import tqdm

# Own Classes
import MANET.Components as net_comps
from MANET.Network import Network

"""
# ExhaustiveSearch.py

To evaluate Agents and Baseline it is beneficial to know what the optimum throughput to be achieved in any scenario would be, regardless of the complexity of its calculations, since this is not a viable method consider to solve the MANET relocation problem. 

Hence, an exhaustive search algo was created which more or less efficiently searches the space with the MANET nodes to maximize throughput. It is based on some intuitive idea, which intuitively promises to find a solution very close to the optimum.

To use it one must create and [ExhausiveSearcher](#OptimumFinder.ExhaustiveSearch.ExhaustiveSearcher)-Object. The one can call [startSearch()](#OptimumFinder.ExhaustiveSearch.ExhaustiveSearcher.startSearch) and anything else will be taken care for. The idea behind making it a class is to simplify keeping the state of shared objects and the updating mechanism of the progress bar which is important. Otherwise, it is impossible to tell how long the search would take which can be very long.

__Intuitive Idea__
Since, in an exhaustive search one can never exactly reach a continuous space level search this is also not tried here. It is also important to know that even if one nodes moves by 1 in the playground of [0,100]^2 the change in throughput will be minimal. Hence a position allocation for the nodes that deviates from the optimum by epsilon in [0,0.5) is acceptable. Hence, the idea is to do a first rough grid search and then a refined search, though limiting the space to search to only the sensible space, as follows:

1. Reduce the space to search to the convex hull of the users
2. Since, if there are jammers the optimum could be slightly outside of that hull extend it by a circle around the jammers position
3. Lay a rather rough grid over the working set (hull +- jammer circles)
4. Find all possible allocations on that set for the nodes
5. Find the top 5 allocations in terms of throughput
6. For those allocations refine the search and check again
"""


class ExhaustiveSearcher:
    """Can do an exhaustive search that finds the optimal solution.

    How it works:

    1. Reduce the space to search to the convex hull of the users
    2. Since, if there are jammers the optimum could be slightly outside of that hull extend it by a circle around the jammers position
    3. Lay a rather rough grid over the working set (hull +- jammer circles)
    4. Find all possible allocations on that set for the nodes
    5. Find the top 5 allocations in terms of throughput
    6. For those allocations refine the search and check again
    """

    def __init__(
        self,
        nodes: np.ndarray[net_comps.MANETNode],
        jammers: np.ndarray[net_comps.Jammer],
        users: np.ndarray[net_comps.User],
        gridResolution: int = 50,
        refinementResolution: int = 20,
        maxLength: int = 3,
    ) -> None:
        """Creates an ExhaustiveSearcher that can do an exhaustive search.

        Args:
            nodes (np.ndarray[net_comps.MANETNode]): nodes whose positions are to be optimized
            jammers (np.ndarray[net_comps.Jammer]): jammers, which impact the desicions
            users (np.ndarray[net_comps.User]): user for which the nodes create the backbone
            gridResolution (int, optional): grid points per axis. Defaults to 50.
            refinementResolution (int, optional): grid points per axis for refined search. Defaults to 20.
            maxLength (int, optional): number of top allocations to consider. Defaults to 3.
        """
        # Flag if solution was found
        self.solutionFound = False

        # Optimal values
        self.best_throughput = None
        self.best_allocation = None

        # Components
        self.nodes = nodes
        self.jammers = jammers
        self.users = users

        # Network
        self.network = Network(self.nodes, self.jammers, self.users)
        error_term = 10
        self.max_possible_throughput = self.network.getMaxThroughput() - error_term

        # Params
        self.gridResolution = gridResolution
        self.refinementResolution = refinementResolution
        self.maxLength = maxLength

        # Progressbar
        self.pbar = tqdm(
            desc="Sampling Allocations",
            colour="#FFA500",
        )
        self.findTopAllocTime = math.comb(self.gridResolution**2, len(self.nodes))
        self.refinementTime = self.maxLength * np.power(
            self.refinementResolution**2, len(self.nodes)
        )
        self._setPbarTotalTimesteps()

    def startSearch(self) -> tuple:
        """Do an exhaustive search that finds the optimal solution.

        How it works:

        1. Reduce the space to search to the convex hull of the users
        2. Since, if there are jammers the optimum could be slightly outside of that hull extend it by a circle around the jammers position
        3. Lay a rather rough grid over the working set (hull +- jammer circles)
        4. Find all possible allocations on that set for the nodes
        5. Find the top 5 allocations in terms of throughput
        6. For those allocations refine the search and check again

        Returns:
            tuple: throughput, optimal positions
        """
        # Array of users' positions
        user_positions = np.array([user.pos for user in self.users])

        # Get Hull
        hull: Polygon = getConvexHull(user_positions)

        # Now, we have limited the space to a smaller one which still includes the optimum
        working_set = self._extendOrReduceHull(hull)
        print("Found working set.")

        # Get grid points over the convex hull
        grid_points = generateGrid(working_set, resolution=self.gridResolution)
        print("Set grid.")

        # Get all position allocations for the nodes assuming they are non-distinguishable
        allocation_generator = allocateNodes(
            grid_points=grid_points, num_nodes=len(self.nodes), distinguishable=False
        )
        print("Found allocations.")

        # Update progress bar
        self.findTopAllocTime = math.comb(len(grid_points), len(self.nodes))
        self._setPbarTotalTimesteps()

        # Find top five allocations
        top_allocations = self.findTopAllocations(allocation_generator)

        if self.solutionFound:
            return self.best_throughput, self.best_allocation

        top_allocations_only = [
            allocation for throughput, allocation in top_allocations
        ]  # exclude throughput

        # For the top allocations to a refined search
        best_throughput, best_allocation = self.refineBestAllocation(
            top_allocations_only,
            working_set,
        )

        self.pbar.close()

        return best_throughput, best_allocation

    def refineBestAllocation(self, top_allocations, working_set) -> tuple:
        """
        Refine the search for the best allocation by conducting a more detailed search
        around the grid points of the top allocations.

        Args:
            top_allocations (list of tuples): Initial top allocations of nodes to positions.
            working_set (Polygon): The polygon representing the working area.

        Returns:
            tuple: The best refined allocation that maximizes throughput.
        """
        best_allocation = self.best_allocation
        best_throughput = self.best_throughput

        # Get bounds of the working set
        minx, miny, maxx, maxy = working_set.bounds
        lenx = maxx - minx
        leny = maxy - miny

        for allocation in top_allocations:
            # Generate a finer grid around each point in the allocation
            refined_points_list = []
            for point in allocation:
                # Define a small bounding box around the point
                refinmentx = lenx / self.gridResolution
                refinmenty = leny / self.gridResolution
                minx, maxx = (
                    point[0] - refinmentx,
                    point[0] + refinmentx,
                )  # +/- refinment_box_length unit in x-direction
                miny, maxy = (
                    point[1] - refinmenty,
                    point[1] + refinmenty,
                )  # +/- refinment_box_length unit in y-direction

                x = np.linspace(minx, maxx, self.refinementResolution)
                y = np.linspace(miny, maxy, self.refinementResolution)
                refined_points = [(xi, yi) for xi in x for yi in y]
                # Filter refined points to keep only those within the working set
                refined_points = [
                    point
                    for point in refined_points
                    if working_set.covers(Point(point))
                ]

                refined_points_list.append(refined_points)

            # Generate new refined allocations
            refined_allocations = self._generateRefinedAllocations(refined_points_list)

            # Update numb of iteration prediction
            self.refinementTime -= np.power(
                self.refinementResolution**2, len(self.nodes)
            ) - len(refined_allocations)
            self._setPbarTotalTimesteps()

            # Evaluate each refined allocation
            for refined_allocation in refined_allocations:
                # Set node positions based on the refined allocation
                for node, pos in zip(self.network.nodes, refined_allocation):
                    node.pos = np.array(pos)

                # Calculate throughput for this allocation
                throughput = self.sampleThroughput()

                # If maximum possible throughput is already found we don't care to further search
                if self._maxThroughputReached(throughput):
                    print("-----Early solution found")
                    return throughput, allocation

                # Update the best allocation if this is better
                if throughput > best_throughput:
                    best_throughput = throughput
                    best_allocation = refined_allocation

        return best_throughput, best_allocation

    def findTopAllocations(self, allocation_generator: Generator) -> list:
        """Finds the top {maxLength} allocations that maximize the throughput.

        Args:
            allocation_generator (Generator): Possible allocations of nodes to positions.

        Returns:
            list: A list of tuples of the top {maxLength} allocations with the throughput e.g (throughput, allocation), sorted by throughput (highest first).
        """
        # List to maintain top allocations
        top_allocations = []

        # Iterate over all allocations
        for allocation in allocation_generator:
            # Set node positions based on the allocation
            for node, pos in zip(self.network.nodes, allocation):
                node: net_comps.MANETNode
                node.pos = np.array(pos)

            # Calculate the throughput for the current allocation
            throughput = self.sampleThroughput()

            # If maximum possible throughput is already found we don't care to further search
            if self._maxThroughputReached(throughput):
                print("----------Found solution early")
                self.best_throughput = throughput
                self.best_allocation = allocation
                return [(throughput, allocation)]

            # Add the allocation and throughput to the list
            if len(top_allocations) < self.maxLength:
                top_allocations.append((throughput, allocation))
            else:
                # Find the minimum throughput in the list
                min_throughput, min_allocation = min(
                    top_allocations, key=lambda x: x[0]
                )
                # Replace the minimum throughput allocation if the current one is better
                if throughput > min_throughput:
                    top_allocations.remove((min_throughput, min_allocation))
                    top_allocations.append((throughput, allocation))

        # Sort the top allocations by throughput in descending order
        top_allocations.sort(reverse=True, key=lambda x: x[0])
        self.best_throughput, self.best_allocation = top_allocations[0]
        return top_allocations

    def sampleThroughput(self) -> float:
        """Sample the throughput by updating the network and letting all users send

        Returns:
            float: throughput value
        """
        self.network.updateNetwork()
        self.network.routeOfferedLoads()
        throughput = self.network.getThroughput()
        self.pbar.update(1)
        return throughput

    def _generateRefinedAllocations(
        self, refined_points_list: list, distinguishable: bool = False
    ) -> list:
        """
        Generate all possible refined allocations of nodes to refined points.

        Args:
            refined_points_list (list of list of tuples): List of refined points for each node.
            distinguishable (bool): whether nodes differ from each other or not

        Returns:
            list: A list of tuples, where each tuple represents an allocation.
                Each tuple has length `num_nodes` and contains refined point coordinates.
        """
        if not distinguishable:
            # Use product to consider all arrangements
            allocations = list(product(*refined_points_list))
        else:
            # Use combinations to consider all subsets without order
            all_permutations = permutations(refined_points_list)

            # Generate Cartesian products for each permutation and combine them
            allocations = []
            for perm in all_permutations:
                allocations.extend(product(*perm))

        return allocations

    def _setPbarTotalTimesteps(self):
        """Update total steps for progressbar"""
        self.pbar.total = self.findTopAllocTime + self.refinementTime

    def _maxThroughputReached(self, curr_throuhgput: int) -> bool:
        """Check if max throughput was reached

        Args:
            curr_throuhgput (int): current throughput value

        Returns:
            bool: whether max throughput has already been reached
        """
        maxThroughputReached = curr_throuhgput >= self.max_possible_throughput
        return maxThroughputReached

    def _extendOrReduceHull(self, hull: Polygon) -> Polygon:
        """Either remove part of the hull or add part of a circle to the hull

        Args:
            hull (Polygon): hull given

        Returns:
            Polygon: updated hull
        """
        # Either remove part of the hull or add part of a circle to the hull
        for jammer in self.jammers:
            jammer: net_comps.Jammer
            # Effective radius
            radius = jammer.approxRadius(1e-6)  # because env noise is 1e-7
            circle = (jammer.pos, radius)

            # If jammer is inside the convex hull it can extend it
            if hull.covers(Point(jammer.pos)):
                new_hull = joinPolygonWithCircle(hull, circle)
            # If jammer is outside the hull it can decrease it
            else:
                new_hull = subtractCircleFromPolygon(hull, circle)

            # Ensure the hull does not become empty
            if not new_hull.is_empty:
                hull = new_hull

        return hull


def getConvexHull(
    points: np.ndarray, thickness: float = 1.0
) -> Union[Polygon, LineString]:
    """
    Get the convex hull as a polygon or line from a set of points.

    Args:
        points (np.ndarray): Points for which one wants the convex hull.
        thickness (float): Thickness to extend the line if only two points are given.

    Returns:
        Polygon or LineString: Convex hull represented as a polygon (or a line if only two points).
    """
    if len(points) == 2:
        # If only two points are given, return a rectangle with the specified thickness
        line = LineString(points)
        return line.buffer(thickness / 2, cap_style=2)  # cap_style=2 for square ends

    # For three or more points, compute the convex hull normally
    hull = ConvexHull(points)
    hull_polygon = Polygon(points[hull.vertices])
    return hull_polygon


def joinPolygonWithCircle(polygon: Polygon, circle: tuple) -> Polygon:
    """Join polygon with Circle

    Args:
        polygon (Polygon): polygon to join with the circle
        circle (tuple): circle to join with polygon

    Returns:
        Polygon: Union of polygon and circle
    """
    center, radius = circle
    circle_polygon: Polygon = Point(center).buffer(radius)
    new_polygon = polygon.union(circle_polygon)

    return new_polygon


def subtractCircleFromPolygon(polygon: Polygon, circle: tuple) -> Polygon:
    """Subtract circle from polygon.

    Args:
        polygon (Polygon): polygon to subtract circle from
        circle (tuple): circle to subtract from polygon

    Returns:
        Polygon: polygon wthout the circle
    """
    center, radius = circle
    circle_polygon: Polygon = Point(center).buffer(radius)
    new_polygon = polygon.difference(circle_polygon)

    return new_polygon


def generateGrid(
    working_set: Union[Polygon, LineString], resolution: int = 100
) -> list:
    """
    Generate grid points within the working set.

    Args:
        working_set (Union[Polygon, LineString]): The working set polygon or line.
        resolution (int): Number of points per axis for the grid.

    Returns:
        list: Array of (x, y) points inside the working set.
    """
    if isinstance(working_set, LineString):
        # Create evenly distributed points along the line
        distances = np.linspace(0, working_set.length, resolution)
        inside_points = [
            tuple(working_set.interpolate(distance).coords[0]) for distance in distances
        ]
    else:
        # Get bounds of the working set
        minx, miny, maxx, maxy = working_set.bounds

        # Create grid points
        x = np.linspace(minx, maxx, resolution)
        y = np.linspace(miny, maxy, resolution)
        grid_points = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)

        # Filter points inside the working set
        inside_points = [
            tuple(point) for point in grid_points if working_set.covers(Point(point))
        ]

    return inside_points


def allocateNodes(grid_points, num_nodes, distinguishable=True) -> Generator:
    """
    Generate all possible allocations of nodes to grid points.

    Args:
        grid_points (array-like): List or array of available grid points.
        num_nodes (int): Number of nodes to allocate.
        distinguishable (bool): If True, nodes are distinguishable (order matters).
                               If False, nodes are not distinguishable (order does not matter).

    Returns:
        Generator: A generator of tuples, where each tuple represents an allocation.
                   Each tuple has length `num_nodes` and contains grid point indices or coordinates.
    """
    if num_nodes > len(grid_points):
        raise ValueError("Number of nodes cannot exceed the number of grid points.")
    if distinguishable:
        # Use permutations to consider all arrangements
        allocation_generator = permutations(grid_points, num_nodes)
    else:
        # Use combinations to consider all subsets without order
        allocation_generator = combinations(grid_points, num_nodes)

    return allocation_generator
