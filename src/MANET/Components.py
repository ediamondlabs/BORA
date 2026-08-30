import numpy as np
from copy import deepcopy
from typing import Union, Type, List, FrozenSet
from collections.abc import Iterable

"""
# Components.py
In the [Components.py](#MANET.Components) the core network components for the simulation are modelled. 
The main functionalities are modelled in [NetworkComponent](#MANET.Components.NetworkComponent) from which all, 
`MANETNode`, `Jammer` and `User` inherit and add their individual functionalities.

__NetworkComponent__

Includes methods for calculating signal transmission, capacity, and component interactions. 
The signal power transmitted is calculated using the simplified `Friis Equation`, 
where equal Gain and additional path loss are assumed. For the transmission capacity, 
a link with capacity being the minimum from the capacity deduced by `Shannon's Equation` and the `Nyquist Theorem` is considered.

Following parameters are used:

- `BANDWIDTH`: 20 MHz (2.4 GHz band channels typically have Bandwidth of 20 MHz or 40 MHz)
- `FREQUENCY`: 2.4 GHz (2.4 GHz band channel)
- `SPEED_OF_LIGHT`: 300,000,000 m/s (Speed of light in m/s)
- `PATH_LOSS_EXPONENT`: 1.8 (Adjust based on environment)
- `ENVIRONMENT_NOISE`: 1e-6 mW (Environmental noise level)

__Components__

The resulting components used in the simulation are:

- [MANETNode](#MANET.Components.MANETNode): Which has the exact same functionality as the `NetworkComponent`.
- [Jammer](#MANET.Components.Jammer): Has additional functions to instead of transmit signal, 
jam components/transmit noise, which the `NetworkComponent` takes as `interference`.
- [User](#MANET.Components.User): A User additionaly, has a `destination` and an `offered load`. 
So, at each time step in the simulation it wants to send an `offered load` to its `destination`, which is another User.
- [ByzantineNode](#MANET.Components.ByzantineNode): A subclass of `MANETNode` that models Byzantine faults. Supports routing-layer attacks (greyhole, sinkhole, selective_jamming, position_spoofing) and observation-layer attacks that falsify the broadcast state vector seen by neighbouring nodes (obs_user_dir, obs_capacity, obs_demand, obs_interference_lie, obs_noise).
"""

SPEED_OF_LIGHT: int = 3 * 1e8  # Speed of light in m/s

ENVIRONMENT_NOISE: float = 1e-6

PATH_LOSS_EXPONENT: int = 1.8  # Adjust based on environment


class NetworkComponent:
    """
    Models the main functionalities of a wireless device needed for the simulation.
    Includes methods for calculating signal transmission, capacity, and component interactions.
    MANETNode, Jammer and User are all NetworkComponents and hence, inherit the basic functionalities of a NetworkComponent.

    Attributes:
        id (int or str): Identification of the network component (e.g. "User 1", 1, "Jammer 2", etc.)
        pos (np.ndarray): Position of the network component 2D.
        signalReceived_mW (float): Signal received in milliwatts.
        interferenceReceived_mW (float): Interference received in milliwatts.
        radius (float): Approximated radius of the network component.
        trafficHosted (int): Amount of traffic a component currently forwards.
        BANDWIDTH (int): Bandwidth of the channel in Hz.
        FREQUENCY (int): Frequency of the channel in Hz.
    """

    BANDWIDTH: int = (
        20 * 1e6
    )  # 2.4 GHz band channels typically have Bandwidth of 20 MHz or 40 MHz.

    FREQUENCY: int = 2.4 * 1e9  # 2.4 GHz  band channel

    def __init__(
        self, id: Union[int, str], pos: np.ndarray, signalPower_mW: float = 1e3
    ) -> None:
        """
        Initializes the NetworkComponent with an ID, position, and signal power.

        Args:
            id (int or str): Identification of the network component (e.g. "User 1", 1, "Jammer 2", etc.)
            pos (np.ndarray): 2D position of the network component.
            signalPower_mW (float): Signal power in mW
        """
        self.id: Union[int, str] = id  # identification (e.g. 1 or 'Node 1')
        self.pos: np.ndarray = pos
        self.signalPower_mW: float = (
            signalPower_mW  # signal power in mW (typically between 500mW and 1W)
        )

        self.signalReceived_mW: float = 0  # signal received in mW
        self.interferenceReceived_mW: float = 0  # interference received in mW
        self.radius: float = self.approxRadius()

        self.trafficHosted: int = 0  # in bits

        # Byzantine robustness signals — computed by Network.computeRobustnessSignals each step.
        # All signals are initialised to "honest" values so that they are safe to read before
        # the first call to computeRobustnessSignals.
        self.signal_consistency: float = 1.0   # 1.0 = honest; <1.0 = sinkhole suspected
        self.pos_consistency: float = 1.0      # 1.0 = honest; <1.0 = position spoofing suspected
        self.unexplained_interference: float = 0.0  # >0 = selective jamming suspected (mW)
        self.trust: float = 1.0               # EMA composite trust; 1.0 = fully trusted

    # * Radius
    def approxRadius(self, capacity_threshold: float = 500e3) -> float:  # 500 kbit
        """
        Approximates the transmission radius by calculating the distance from which on an other
        Network Component receives less capacity then, the capacity_threshold demands.

        Args:
            capacity_threshold (float, optional): Defaults to 500 kbit.

        Returns:
            float: Approximate radius in meters
        """
        # Calculate wavelength
        wavelength = SPEED_OF_LIGHT / self.FREQUENCY

        # Minimum required SNR based on Shannon capacity
        min_snr = 2 ** (capacity_threshold / self.BANDWIDTH) - 1

        # Required transmission power in mW
        required_power_mW = ENVIRONMENT_NOISE * min_snr

        # Calculate approximate radius using path loss model
        approx_radius = np.float_power(
            (self.signalPower_mW * (wavelength / (4 * np.pi)) ** 2 / required_power_mW),
            1 / (2 + PATH_LOSS_EXPONENT),
        )

        return approx_radius

    def reaches(self, otherComp: "NetworkComponent") -> bool:
        """
        Determines if this component can reach another component based on distance between them and
        transmitting component's transmission radius. In other words, a component reaches another one, if
        it is in its transmission radius.

        Args:
            otherComp (NetworkComponent): The other network component to check reachability.

        Returns:
            bool: True if the other component is within reach, False otherwise.
        """
        distance = np.linalg.norm(self.pos - otherComp.pos)
        reaches: bool = distance <= self.radius

        return reaches

    # * Signal transmission
    def calcSignalTransmitted(self, otherComp: "NetworkComponent") -> float:
        """
        Calculates the signal this component transmits to another component according to the
        Friis Equation with equal Gain assumed and considering a path loss exponent.

        Args:
            otherComp (NetworkComponent): Other component set to receive signal.

        Returns:
            float: Transmitted signal power in mW.
        """
        epsilon = 1e-6  # Prevent divide by zero
        distance = np.linalg.norm(self.pos - otherComp.pos) + epsilon

        signalTransmitted: float = 0.0

        if self.reaches(otherComp):
            wavelength = SPEED_OF_LIGHT / self.FREQUENCY
            transmittedSignal = (
                self.signalPower_mW
                * (wavelength / (4 * np.pi)) ** 2
                * np.float_power(distance, -(2 + PATH_LOSS_EXPONENT))
            )
            signalTransmitted = min(self.signalPower_mW, transmittedSignal)

        return signalTransmitted

    def transmitSignal(self, networkComps: Iterable) -> None:
        """
        Transmits the signal to other components.

        Args:
            networkComps (NetworkComponent or iterable of NetworkComponent): The network component(s) to transmit signal to.
        """
        if isinstance(networkComps, Iterable) and not isinstance(
            networkComps, (str, bytes)
        ):
            components = networkComps
        else:
            components = [networkComps]

        for otherComp in components:
            self._transmit_to_component(otherComp)

    def _transmit_to_component(self, otherComp: "NetworkComponent") -> None:
        """
        Transmit signal to another component

        Args:
            otherComp (NetworkComponent): The network component to transmit signal to.
        """
        if self != otherComp:
            signalTransmitted = self.calcSignalTransmitted(otherComp)
            otherComp._receiveSignal(signalTransmitted)

    def _receiveSignal(self, signal: float) -> None:
        """
        Receive signal

        Args:
            signal (float): Signal power received in mW.
        """
        self.signalReceived_mW += signal

    # * Capacity
    def calcCapacity(self, otherComp: "NetworkComponent") -> int:
        """
        Calculates the communication capacity of a link to the other component.

        Args:
            otherComp (NetworkComponent): The other network component.

        Returns:
            int: Communication capacity in bits per second.
        """
        signalTransmitted_mW = self.calcSignalTransmitted(otherComp)

        C = int(
            shannonTheorem(
                signalReceived_mW=signalTransmitted_mW,
                interferenceReceived_mW=otherComp.interferenceReceived_mW,
                noiseReceived_mW=ENVIRONMENT_NOISE,
                BANDWIDTH=self.BANDWIDTH,
            )
        )

        return C

    # * Traffic
    def hostTraffic(self, traffic: int) -> None:
        """
        Adds the traffic hosted to its memory.

        Args:
            traffic (int): Traffic (in bit) that has been forwarded over this node.
        """
        self.trafficHosted += traffic

    # * Reset values
    def resetSignalReceived(self) -> None:
        """
        Resets the signal power received by the network component.
        """
        self.signalReceived_mW = 0

    def resetInterferenceReceived(self) -> None:
        """
        Resets the total interference received.
        """
        self.interferenceReceived_mW = 0

    def resetTrafficHosted(self) -> None:
        """
        To reset the traffic hosted value. (E.g. in Enviroments after each step.)
        """
        self.trafficHosted = 0

    def resetAll(self) -> None:
        """
        Resets the total noise received, the total signal power received and the total traffic hosted by the network component.
        """
        self.resetInterferenceReceived()
        self.resetSignalReceived()
        self.resetTrafficHosted()

    # * Dict or not to dict
    def to_dict(self) -> dict:
        """Convert to a dictionary representation of the object

        Returns:
            dict: dictionary representation of the current object
        """
        return {
            "id": self.id,
            "pos": self.pos.tolist(),
            "signalPower_mW": float(self.signalPower_mW),
            "signalReceived_mW": float(self.signalReceived_mW),
            "interferenceReceived_mW": float(self.interferenceReceived_mW),
            "radius": float(self.radius),
            "trafficHosted": int(self.trafficHosted),
            "BANDWIDTH": int(self.BANDWIDTH),
            "FREQUENCY": int(self.FREQUENCY),
            "SPEED_OF_LIGHT": int(SPEED_OF_LIGHT),
            "PATH_LOSS_EXPONENT": int(PATH_LOSS_EXPONENT),
            "ENVIRONMENT_NOISE": float(ENVIRONMENT_NOISE),
        }

    @classmethod
    def from_dict(cls: Type["NetworkComponent"], data: dict) -> "NetworkComponent":
        """Recreate a network component from its dictionary representation.

        Args:
            cls (Type[NetworkComponent]): The network component constructor.
            data (dict): Network component represented as dicitonary

        Returns:
            NetworkComponent: Recreated network component object
        """
        # Create an instance of itself
        instance: "NetworkComponent" = cls(
            id=data["id"],
            pos=np.array(data["pos"]),  # Convert list back to np.array
            signalPower_mW=data["signalPower_mW"],
        )

        # Set the attributes
        instance.signalReceived_mW = data["signalReceived_mW"]
        instance.interferenceReceived_mW = data["interferenceReceived_mW"]
        instance.radius = data["radius"]
        instance.trafficHosted = data["trafficHosted"]

        return instance

    # * Special methods
    def __repr__(self) -> str:
        """
        Returns a string representation of the NetworkComponent.

        Returns:
            str: String representation of the NetworkComponent.
        """
        return f"{self.__class__.__name__}(id={self.id}, pos={self.pos}, signalPower_mW={self.signalPower_mW})"

    def __eq__(self, other: "NetworkComponent") -> bool:
        """
        Checks if two NetworkComponent instances are equal, which they are if they are of the same type and have the same id.

        Args:
            other (NetworkComponent): The other network component to compare with.

        Returns:
            bool: True if both instances are equal, False otherwise.
        """
        if not isinstance(other, self.__class__):
            return False
        elif self.id != other.id:
            return False

        return True

    def __hash__(self) -> int:
        """
        Returns the hash of the network component based on its ID.

        Returns:
            int: Hash of the ID.
        """
        return hash(self.id)


class MANETNode(NetworkComponent):
    """
    Models a Mobile Ad-hoc Network (MANET) node, which inherits its functionality
    from `NetworkComponent`.

    Attributes:
        id (int or str): Identification of the network component (e.g. "User 1", 1, "Jammer 2", etc.)
        pos (np.ndarray): Position of the network component 2D.
        signalReceived_mW (float): Signal received in milliwatts.
        interferenceReceived_mW (float): Interference received in milliwatts.
        radius (float): Approximated radius of the network component.
        trafficHosted (int): Amount of traffic a component currently forwards.
        BANDWIDTH (int): Bandwidth of the channel in Hz.
        FREQUENCY (int): Frequency of the channel in Hz.
    """

    @classmethod
    def from_dict(cls: Type["MANETNode"], data: dict) -> "MANETNode":
        """Recreate a MANETNode object from its dictionary representation.

        Args:
            cls (Type[MANETNode]): The MANETNode constructor.
            data (dict): Representation of the MANETNode object.

        Returns:
            MANETNode: Recreated MANETNode object.
        """
        instance = super(MANETNode, cls).from_dict(data)
        return instance

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result.__init__(
            id=deepcopy(self.id, memo),
            pos=deepcopy(self.pos, memo),
            signalPower_mW=deepcopy(self.signalPower_mW, memo),
        )
        # Copy parent attributes
        result.signalReceived_mW = deepcopy(self.signalReceived_mW, memo)
        result.interferenceReceived_mW = deepcopy(self.interferenceReceived_mW, memo)
        result.radius = deepcopy(self.radius, memo)
        result.trafficHosted = deepcopy(self.trafficHosted, memo)
        return result

class ByzantineNode(MANETNode):
    """
    Models a Byzantine Mobile Ad-hoc Network (MANET) node, which inherits its functionality
    from `MANETNode`.

    A `ByzantineNode` can perform one or more attacks simultaneously. Pass a single string
    or a list of strings to `attack_type`. All active attack behaviours are executed each step.

    Routing-layer attacks (affect traffic and physics):

    - "greyhole": Silently drops a configurable fraction (`drop_rate`) of traffic routed
      through this node. Applied post-routing in `Network.updateByzantineEffect()`.
    - "sinkhole": Advertises inflated link capacities and signal-power observations to attract
      more traffic, then drops it. The routing solver sees `capacity_inflation_factor`× the real
      capacity; the observation wrapper reads `observed_signal_power_mW` / `observed_traffic_hosted`.
    - "selective_jamming": Acts as an insider jammer. Each network step `Network.updateByzantineJamming()`
      calls `jamNeighbors()`, which raises `interferenceReceived_mW` on all reachable components,
      degrading their Shannon capacity.
    - "position_spoofing": Reports a fake position to the observation wrapper via `observed_pos`.
      Physics (signal propagation, routing) use the real `pos`; only the RL agent's state is corrupted.

    Observation-layer attacks (falsify the broadcast state vector seen by neighbouring nodes):

    - "obs_user_dir": Adds a fixed 2-D offset to the reported closest-user direction vectors
      (`observed_user_dir_offset`). Neighbours receive wrong directions to traffic demand.
      Undetectable via physics — no cross-check for another node's closest user is possible.
    - "obs_capacity": Scales reported in/out capacities by `obs_capacity_factor`. Values > 1
      attract neighbours; values < 1 repel them. Partially detectable via signal cross-check.
    - "obs_demand": Scales reported sender/receiver demand by `obs_demand_factor`. Values > 1
      fabricate high demand near this node ("demand sinkhole"). Completely undetectable.
    - "obs_interference_lie": Scales reported received interference by `obs_interference_factor`.
      Factor 0.0 hides all interference, luring neighbours into a jammed zone. Completely undetectable.
    - "obs_noise": Adds zero-mean Gaussian noise (std `obs_noise_std`) to the full broadcast
      state slice as seen by neighbours. Maximally disruptive but also maximally detectable.

    Multiple attacks can be combined, e.g. `attack_type=["obs_demand", "obs_interference_lie"]`.

    New attack types can be added by extending `VALID_ATTACK_TYPES` and implementing the
    corresponding logic here and in `ObservationWrappers.py`.

    Attributes:
        attack_type (str or List[str]): The attack type(s) as originally provided.
        attack_types (FrozenSet[str]): Normalised set of active attack types used for all internal checks.
        drop_rate (float): Fraction of traffic to drop for greyhole/sinkhole (0.0 to 1.0).
        capacity_inflation_factor (float): Multiplier applied to reported link capacity for sinkhole.
        jamming_power_factor (float): Multiplier on signalPower_mW used as interference for selective_jamming.
        pos_offset (np.ndarray): 2-D position offset added to the real position for position_spoofing.
        obs_user_dir_offset (np.ndarray): 2-D offset added to reported closest-user direction for obs_user_dir.
        obs_capacity_factor (float): Multiplier on reported in/out capacities for obs_capacity.
        obs_demand_factor (float): Multiplier on reported sender/receiver demand for obs_demand.
        obs_interference_factor (float): Multiplier on reported received interference for obs_interference_lie.
        obs_noise_std (float): Std dev of zero-mean Gaussian noise added to broadcast state for obs_noise.
        is_byzantine (bool): Always True, used for identification.
    """

    VALID_ATTACK_TYPES = {
        "greyhole", "sinkhole", "selective_jamming", "position_spoofing",
        "obs_user_dir", "obs_capacity", "obs_demand", "obs_interference_lie", "obs_noise",
        "obs_capacity_deflate", "obs_demand_deflate", "obs_replay", "obs_full",
    }

    def __init__(
        self,
        id,
        pos,
        signalPower_mW=1e3,
        attack_type: Union[str, List[str]] = "greyhole",
        drop_rate: float = 0.5,
        capacity_inflation_factor: float = 3.0,
        jamming_power_factor: float = 1.0,
        pos_offset: np.ndarray = None,
        obs_user_dir_offset: np.ndarray = None,
        obs_capacity_factor: float = 1.0,
        obs_demand_factor: float = 1.0,
        obs_interference_factor: float = 1.0,
        obs_noise_std: float = 0.0,
        obs_capacity_deflate_factor: float = 1.0,
        obs_demand_deflate_factor: float = 1.0,
        obs_replay_delay: int = 0,
    ):
        super().__init__(id, pos, signalPower_mW)

        # Normalise to a list for validation, then store as frozenset for O(1) membership tests
        types_list = [attack_type] if isinstance(attack_type, str) else list(attack_type)
        invalid = set(types_list) - self.VALID_ATTACK_TYPES
        if invalid:
            raise ValueError(
                f"Invalid attack_type(s) {invalid}. "
                f"Must be a subset of {self.VALID_ATTACK_TYPES}."
            )

        self.attack_type = attack_type          # preserved for serialisation / logging
        self.attack_types: FrozenSet[str] = frozenset(types_list)
        self.is_byzantine = True

        # Routing-layer attack parameters
        self.drop_rate = drop_rate
        self.capacity_inflation_factor = capacity_inflation_factor
        self.jamming_power_factor = jamming_power_factor
        self.pos_offset = pos_offset if pos_offset is not None else np.zeros(2)

        # Observation-layer attack parameters
        self.obs_user_dir_offset = obs_user_dir_offset if obs_user_dir_offset is not None else np.zeros(2)
        self.obs_capacity_factor = obs_capacity_factor
        self.obs_demand_factor = obs_demand_factor
        self.obs_interference_factor = obs_interference_factor
        self.obs_noise_std = obs_noise_std
        self.obs_capacity_deflate_factor = obs_capacity_deflate_factor
        self.obs_demand_deflate_factor = obs_demand_deflate_factor
        self.obs_replay_delay = obs_replay_delay
        # Set per-step by ByzantineAttackerEnv when attack_type="obs_full".
        self.obs_full_offset: np.ndarray | None = None

    # ---------------------------------------------------------------------------
    # Greyhole / Sinkhole — traffic dropping
    # ---------------------------------------------------------------------------

    def getDropped(self, traffic: float) -> float:
        """
        Calculates the amount of traffic dropped by this Byzantine node.
        Used by greyhole and sinkhole attack types.

        Args:
            traffic (float): The traffic routed through this node.

        Returns:
            float: The amount of traffic dropped.
        """
        if self.attack_types & {"greyhole", "sinkhole"}:
            return traffic * self.drop_rate
        return 0.0

    # ---------------------------------------------------------------------------
    # Sinkhole — inflated capacity / observation spoofing
    # ---------------------------------------------------------------------------

    def calcCapacity(self, otherComp: "NetworkComponent") -> int:
        """
        Returns the link capacity to `otherComp`. For sinkhole nodes the reported
        capacity is inflated by `capacity_inflation_factor` so the routing solver
        preferentially routes traffic through this node.

        Args:
            otherComp (NetworkComponent): The other network component.

        Returns:
            int: (Possibly inflated) communication capacity in bits per second.
        """
        real_capacity = super().calcCapacity(otherComp)
        if "sinkhole" in self.attack_types:
            return int(real_capacity * self.capacity_inflation_factor)
        return real_capacity

    @property
    def observed_signal_power_mW(self) -> float:
        """Reported signal power seen by the observation wrapper.
        Sinkhole nodes advertise an inflated value to attract the RL agent."""
        if "sinkhole" in self.attack_types:
            return self.signalPower_mW * self.capacity_inflation_factor
        return self.signalPower_mW

    @property
    def observed_traffic_hosted(self) -> int:
        """Reported traffic hosted seen by the observation wrapper.
        Sinkhole nodes inflate this metric to attract relay positioning."""
        if "sinkhole" in self.attack_types:
            return int(self.trafficHosted * self.capacity_inflation_factor)
        return self.trafficHosted

    # ---------------------------------------------------------------------------
    # Selective jamming — insider interference
    # ---------------------------------------------------------------------------

    def jamNeighbors(self, networkComps: Iterable) -> None:
        """
        Emits interference towards all reachable network components.
        Called each step by `Network.updateByzantineJamming()` for selective_jamming nodes.

        Args:
            networkComps (Iterable): All components (nodes + users) that can be jammed.
        """
        for comp in networkComps:
            if comp is not self and self.reaches(comp):
                comp.interferenceReceived_mW += self.signalPower_mW * self.jamming_power_factor

    # ---------------------------------------------------------------------------
    # Position spoofing — fake position for the observation wrapper
    # ---------------------------------------------------------------------------

    @property
    def observed_pos(self) -> np.ndarray:
        """Position reported to the observation wrapper.
        Position-spoofing nodes return their real position shifted by `pos_offset`."""
        if "position_spoofing" in self.attack_types:
            return self.pos + self.pos_offset
        return self.pos

    # ---------------------------------------------------------------------------
    # Observation-layer attacks — falsify broadcast state seen by neighbours
    # ---------------------------------------------------------------------------

    @property
    def observed_user_dir_offset(self) -> np.ndarray:
        """Per-node direction offset injected into closest-user direction getters.
        obs_user_dir nodes return a non-zero offset; honest nodes return None (no injection)."""
        if "obs_user_dir" in self.attack_types:
            return self.obs_user_dir_offset
        return None

    @property
    def observed_capacity_factor(self) -> float:
        """Capacity scaling factor injected into in/out capacity getters.
        obs_capacity nodes return their configured factor; honest nodes return 1.0."""
        if "obs_capacity" in self.attack_types:
            return self.obs_capacity_factor
        if "obs_capacity_deflate" in self.attack_types:
            return self.obs_capacity_deflate_factor
        return 1.0

    @property
    def observed_demand_factor(self) -> float:
        """Demand scaling factor injected into sender/receiver demand getters.
        obs_demand nodes return their configured factor; honest nodes return 1.0."""
        if "obs_demand" in self.attack_types:
            return self.obs_demand_factor
        if "obs_demand_deflate" in self.attack_types:
            return self.obs_demand_deflate_factor
        return 1.0

    @property
    def observed_interferenceReceived_mW(self) -> float:
        """Interference reported to the observation wrapper.
        obs_interference_lie nodes scale their real interference by `obs_interference_factor`."""
        if "obs_interference_lie" in self.attack_types:
            return self.interferenceReceived_mW * self.obs_interference_factor
        return self.interferenceReceived_mW

    @property
    def observed_obs_noise_std(self) -> float:
        """Std dev of zero-mean Gaussian noise added to the full broadcast state slice.
        obs_noise nodes return their configured std dev; honest nodes return 0.0."""
        if "obs_noise" in self.attack_types:
            return self.obs_noise_std
        return 0.0

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result.__init__(
            id=deepcopy(self.id, memo),
            pos=deepcopy(self.pos, memo),
            signalPower_mW=deepcopy(self.signalPower_mW, memo),
            attack_type=deepcopy(self.attack_type, memo),
            drop_rate=deepcopy(self.drop_rate, memo),
            capacity_inflation_factor=deepcopy(self.capacity_inflation_factor, memo),
            jamming_power_factor=deepcopy(self.jamming_power_factor, memo),
            pos_offset=deepcopy(self.pos_offset, memo),
            obs_user_dir_offset=deepcopy(self.obs_user_dir_offset, memo),
            obs_capacity_factor=deepcopy(self.obs_capacity_factor, memo),
            obs_demand_factor=deepcopy(self.obs_demand_factor, memo),
            obs_interference_factor=deepcopy(self.obs_interference_factor, memo),
            obs_noise_std=deepcopy(self.obs_noise_std, memo),
            obs_capacity_deflate_factor=deepcopy(self.obs_capacity_deflate_factor, memo),
            obs_demand_deflate_factor=deepcopy(self.obs_demand_deflate_factor, memo),
            obs_replay_delay=deepcopy(self.obs_replay_delay, memo),
        )
        result.signalReceived_mW = deepcopy(self.signalReceived_mW, memo)
        result.interferenceReceived_mW = deepcopy(self.interferenceReceived_mW, memo)
        result.radius = deepcopy(self.radius, memo)
        result.trafficHosted = deepcopy(self.trafficHosted, memo)
        return result


class Jammer(NetworkComponent):
    """
    Models a jammer device that can interfere with other network components by sending noise.
    Jammer inherits its functionality from `NetworkComponent`.

    Attributes:
        id (int or str): Identification of the network component (e.g. "User 1", 1, "Jammer 2", etc.).
        pos (np.ndarray): Position of the network component in 2D.
        signalReceived_mW (float): Signal received in milliwatts.
        interferenceReceived_mW (float): Interference received in milliwatts.
        radius (float): Approximate radius of the network component.
        trafficHosted (int): Amount of traffic hosted in bits.
        isActive (bool): Whether the jammer is jamming or not.
        BANDWIDTH (int): Bandwidth of the channel in Hz.
        FREQUENCY (int): Frequency of the channel in Hz.
    """

    def __init__(
        self, id: Union[int, str], pos: np.ndarray, signalPower_mW: float = 1e3
    ) -> None:
        """
        Initializes the Jammer with an ID, position, and signal power.

        Args:
            id (int or str): Identification of the jammer.
            pos (np.ndarray): Position of the jammer.
            signalPower_mW (float): Signal power.
        """
        super().__init__(id, pos, signalPower_mW)
        self.isActive: bool = False

    def startJamming(self) -> None:
        """
        Allows the jammer to start jamming. Hence, `jamComps()` will actually jam other
        NetworkComponents.
        """
        self.isActive = True

    def stopJamming(self) -> None:
        """
        Prevents the jammer from jamming. Hence, `jamComps()` will not jam other
        NetworkComponents.
        """
        self.isActive = False

    def jamComps(self, networkComps: np.ndarray) -> None:
        """
        Jams all network components in the given array if the jammer is active.

        Args:
            networkComps (np.ndarray): Array of network components to jam.
        """
        if self.isActive:
            if isinstance(networkComps, Iterable) and not isinstance(
                networkComps, (str, bytes)
            ):
                components = networkComps
            else:
                components = [networkComps]

            for otherComp in components:
                self._jamComp(otherComp)

    def _jamComp(self, networkComp: "NetworkComponent") -> None:
        """
        Jams a specific network component.

        Args:
            networkComp (NetworkComponent): The component to be jammed.
        """
        if self != networkComp:
            interference = self.calcSignalTransmitted(networkComp)
            networkComp.interferenceReceived_mW += interference

    # * Dict or not to dict
    def to_dict(self) -> dict:
        data = super().to_dict()
        data["isActive"] = self.isActive
        return data

    @classmethod
    def from_dict(cls: Type["Jammer"], data: dict) -> "Jammer":
        """Recreate a Jammer object from its dictionary representation

        Args:
            cls (Type[Jammer]): The jammer class constructor.
            data (dict): Dictionary representation of the jammer object.

        Returns:
            Jammer: The recreated jammer object.
        """
        instance: "Jammer" = super(Jammer, cls).from_dict(data)
        instance.isActive = data["isActive"]
        return instance

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result.__init__(
            id=deepcopy(self.id, memo),
            pos=deepcopy(self.pos, memo),
            signalPower_mW=deepcopy(self.signalPower_mW, memo),
        )
        result.signalReceived_mW = deepcopy(self.signalReceived_mW, memo)
        result.interferenceReceived_mW = deepcopy(self.interferenceReceived_mW, memo)
        result.radius = deepcopy(self.radius, memo)
        result.trafficHosted = deepcopy(self.trafficHosted, memo)
        result.isActive = deepcopy(self.isActive, memo)
        return result


class User(NetworkComponent):
    """
    Models a user device in the network which has an offered load and
    a destination, so a recipient of that offered load. A User is a `NetworkComponent`.

    Attributes:
        id (int or str): Identification of the network component (e.g. "User 1", 1, "Jammer 2", etc.).
        pos (np.ndarray): Position of the network component in 2D.
        signalReceived_mW (float): Signal received in milliwatts.
        interferenceReceived_mW (float): Interference received in milliwatts.
        radius (float): Approximate radius of the network component.
        trafficHosted (int): Amount of traffic hosted in bits.
        offeredLoad (int): The amount of data desired to send.
        dest (User): The user which should receive data.
        prevDataDelivered (int): Previous data able to send to destination. Hence, prevDataDelivered <= offeredLoad.
        BANDWIDTH (int): Bandwidth of the channel in Hz.
        FREQUENCY (int): Frequency of the channel in Hz.
        M (int): Number of signal levels.
    """

    def __init__(
        self,
        id: Union[int, str],
        pos: np.ndarray,
        signalPower_mW: float = 1e3,
        offeredLoad: int = 20e6,
        dest: "User" = None,
    ) -> None:
        """
        Initializes the User with an ID, position, signal power, and offered load.

        Args:
            id (int or str): Identification of the user.
            pos (np.ndarray): Position of the user.
            signalPower_mW (float): Signal power in milliwatts. Defaults to 1 Watt.
            offeredLoad (int): Offered load in bits. Defaults to 20 Mb
            dest (User): Other user that offered load should be sent to. Defaults to None.
        """
        super().__init__(id, pos, signalPower_mW)
        self.offeredLoad: int = offeredLoad
        self.dest: "User" = dest
        self.prevDataDelivered: int = 0

    def setOfferedLoad(self, offeredLoad: int) -> None:
        """
        Sets the offered load of the user.

        Args:
            offeredLoad (int): Offered load in bits per second.
        """
        self.offeredLoad = offeredLoad

    def setDestination(self, dest: "User") -> None:
        """
        Sets the destination of the user.

        Args:
            dest (User): Destination user.
        """
        if dest != self:
            self.dest = dest

    def setDataDelivered(self, dataDelivered: int) -> None:
        """Set the data delivered

        Args:
            dataDelivered (int): Data the User was able to send. Always less or equal to the offered load.
        """
        self.prevDataDelivered = dataDelivered

    def resetDataDelivered(self) -> None:
        """
        Reset the data delivered
        """
        self.prevDataDelivered = 0

    def hasDest(self) -> bool:
        """Whether the user currently has a destination

        Returns:
            bool: Whether the user currently has a destination
        """
        return isinstance(self.dest, User)

    # * Overwrite
    def resetAll(self) -> None:
        """
        Reset too the data delivered
        """
        super().resetAll()
        self.resetDataDelivered()

    def to_dict(self) -> dict:
        """Create a dictionary representation of the User object.

        Returns:
            dict: Representation of the User object.
        """
        data = super().to_dict()
        data.update(
            {
                "offeredLoad": self.offeredLoad,
                "dest_id": self.dest.id if self.dest else None,
            }
        )
        return data

    @classmethod
    def from_dict(cls: Type["User"], data: dict, user_lookup: dict = {}) -> "User":
        """Recreate a User object from its dictionary representation.

        Args:
            cls (Type[User]): User class constructor.
            data (dict): User representation.
            user_lookup (dict, optional): Dictionary to find other User objects in order to correctly set the destination User object. Defaults to {}.

        Returns:
            User: The recreated user object.
        """
        instance: "User" = super(User, cls).from_dict(data)
        instance.offeredLoad = data["offeredLoad"]
        instance.dest = user_lookup.get(data["dest_id"]) if data["dest_id"] else None
        return instance

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        # Initialize without dest to avoid recursion issues
        result.__init__(
            id=deepcopy(self.id, memo),
            pos=deepcopy(self.pos, memo),
            signalPower_mW=deepcopy(self.signalPower_mW, memo),
            offeredLoad=deepcopy(self.offeredLoad, memo),
            dest=None,  # Set dest later
        )
        result.signalReceived_mW = deepcopy(self.signalReceived_mW, memo)
        result.interferenceReceived_mW = deepcopy(self.interferenceReceived_mW, memo)
        result.radius = deepcopy(self.radius, memo)
        result.trafficHosted = deepcopy(self.trafficHosted, memo)
        result.prevDataDelivered = deepcopy(self.prevDataDelivered, memo)
        # Deep copy dest, handling potential None or recursive User
        result.dest = deepcopy(self.dest, memo) if self.dest else None
        return result


def shannonTheorem(
    signalReceived_mW: float,
    interferenceReceived_mW: float,
    noiseReceived_mW: float,
    BANDWIDTH: int,
) -> float:
    """
    Calculates the maximum data rate based on the Shannon theorem
    considering the effects of interference and noise.

    Args:
        signalReceived_mW (float): Power of the received signal in milliwatts.
        interferenceReceived_mW (float): Power of the received interference in milliwatts.
        noiseReceived_mW (float): Power of the received noise in milliwatts.
        BANDWIDTH (int): Bandwidth of the channel in Hz.

    Returns:
        float: Maximum data rate in bits per second considering interference and noise.
    """
    C = BANDWIDTH * np.log2(
        1 + (signalReceived_mW / (interferenceReceived_mW + noiseReceived_mW))
    )

    return C


def nyquistTheorem(BANDWIDTH: int, M: int = 8) -> int:
    """
    Calculates the maximum data rate based on the Nyquist theorem
    for a noise-free channel.

    Args:
        BANDWIDTH (int): Bandwidth of the channel in Hz.
        M (int): Number of signal levels per symbol (e.g., 2 for Binary, 4 for QPSK, etc.).

    Returns:
        int: Maximum data rate in bits per second for a noise-free channel.

    Raises:
        ValueError: If M is less than 2.
    """
    if M < 2:
        raise ValueError("M must be at least 2 to represent multiple signal levels.")

    nyquist_capacity = 2 * BANDWIDTH * np.log2(M)
    return int(nyquist_capacity)


