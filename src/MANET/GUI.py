import tkinter as tk
import numpy as np
import queue

from .Components import User

"""
# GUI.py
A simple and fast running GUI build with [tkinter](https://docs.python.org/3/library/tkinter.html). Components are displayed as dots with colors differentiating whether they are MANETNode, Jammer or User. A "connection" is displayed with a directed link/arrow. Although, in Wifi it is less a link but a possible communication route. The `GUI` if closed can't be reopened. However, it can be minimized and maximized without problem and minimizing will significantly increase the simulation's speed.
"""


class MANETGUI(tk.Tk):
    """
    GUI of MANETEnv from MANETEnv.py

    MANETNodes = Blue
    Users = Green
    Jammers = Red
    Connections = Black lines, which“Can be either omni- or bidirectional

    Attributes:
        env (object): The environment object containing the simulation data.
        gui_update_queue (queue.Queue): Queue to receive GUI update commands.
        continue_updating (bool): Flag to control the continuous updating of the GUI.
        speedFactor (float): Factor to control the speed of the simulation.
        NORM_SPEED (int): Normal speed value for the simulation.
        speed (int): Calculated speed based on the speed factor.
        canvas_size (int): Size of the canvas for the visualization.
        scale (float): Scale factor for the canvas.
        paned_window (tk.PanedWindow): Paned window to hold the canvas and the info panel.
        canvas (tk.Canvas): Canvas for drawing the entities and connections.
        info_panel (tk.Frame): Frame for displaying the info panel.
        throughput_label (tk.Label): Label to display the network throughput.
        pause_button (tk.Button): Button to pause and unpause the simulation.
        info_frame (tk.Frame): Frame for displaying packet sending info.
        packet_info_labels (dict): Dictionary to store labels for packet sending info.
        nodes (list): List to store node objects on the canvas.
        node_texts (list): List to store text objects for nodes on the canvas.
        jammers (list): List to store jammer objects on the canvas.
        jammer_texts (list): List to store text objects for jammers on the canvas.
        users (list): List to store user objects on the canvas.
        user_texts (list): List to store text objects for users on the canvas.
        edges (list): List to store edge objects on the canvas.
    """

    def __init__(self, gui_update_queue: queue.Queue, env):
        """
        Initialize the MANETGUI class.

        Args:
            gui_update_queue (queue.Queue): Queue to receive GUI update commands.
            env (object): The environment object containing the simulation data.
        """
        super().__init__()
        self.title("MANET Visualization")

        # Environment
        self.env = env

        # Update queue
        self.gui_update_queue: queue.Queue = gui_update_queue

        # Updates
        self.continue_updating: bool = True
        self.speedFactor: float = 1 / 50
        self.NORM_SPEED = 1
        self.speed = int(self.NORM_SPEED / self.speedFactor)

        # Configure canvas
        self.canvas_size = 500
        self.scale = self.canvas_size / 100

        # Create a PanedWindow to hold the canvas and the info panel
        self.paned_window = tk.PanedWindow(self, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=1)

        # Add the canvas to the PanedWindow
        self.canvas = tk.Canvas(
            self.paned_window,
            width=self.canvas_size,
            height=self.canvas_size,
            bg="white",
        )
        self.paned_window.add(self.canvas)

        # Create the info panel
        self.info_panel = tk.Frame(self.paned_window, width=200)
        self.paned_window.add(self.info_panel)

        # Create the packet sending info
        self.info_frame = tk.Frame(self.info_panel)
        self.info_frame.pack(fill=tk.BOTH, expand=1, pady=10)
        self.packet_info_labels = {}
        self.create_packet_sending_info()

        # Create legend
        self.create_legend()

        # Create network throughput label
        self.throughput_label = tk.Label(self, text="Network Throughput: 0")
        self.throughput_label.pack(side=tk.BOTTOM, pady=10)

        self.last_ep_step_update = -1
        self.ep_step_update_interval = 25  # Update every 25 steps

        # Create speed control buttons
        self.create_speed_buttons()

        # Initialize component lists
        self.nodes = []
        self.node_texts = []
        self.jammers = []
        self.jammer_texts = []
        self.users = []
        self.user_texts = []
        self.edges = []

        # Create initial entities
        self.create_entities(*self.gui_update_queue.get())
        self.update_gui()

        self.protocol("WM_DELETE_WINDOW", self.on_closing)  # Handle window close event
        self.bind("<Unmap>", self.on_minimize)
        self.bind("<Map>", self.on_enlarge)

    def resetComps(
        self,
        node_positions: np.array,
        jammer_positions: np.array,
        user_positions: np.array,
    ) -> None:
        """
        Reset all components by clearing and recreating them based on new data from the queue.
        """
        self.remove_components()
        self.create_packet_sending_info()  # Recreate info panel for new users
        self.create_entities(node_positions, jammer_positions, user_positions)

    def changeSpeed(self, speedFactor: float):
        """
        Change the speed of the simulation.

        Args:
            speedFactor (float): Factor to control the speed of the simulation.
        """
        self.speedFactor = speedFactor
        self.calcSpeed()

    def calcSpeed(self):
        """
        Calculate the speed based on the speed factor.
        """
        self.speed = int(np.ceil(self.NORM_SPEED / self.speedFactor))

    def create_legend(self):
        """
        Create a legend to explain the colors used in the visualization.
        """
        legend_frame = tk.Frame(self)
        legend_frame.pack(side=tk.BOTTOM, pady=10)

        # Create legend items
        self.create_legend_item(legend_frame, "User", "green")
        self.create_legend_item(legend_frame, "MANETNode", "blue")
        self.create_legend_item(legend_frame, "Jammer", "red")

    def create_legend_item(self, parent, text, color):
        """
        Create a single legend item.

        Args:
            parent (tk.Widget): The parent widget to place the legend item in.
            text (str): The label text for the legend item.
            color (str): The color of the dot for the legend item.
        """
        frame = tk.Frame(parent)
        frame.pack(side=tk.LEFT, padx=10)

        canvas = tk.Canvas(frame, width=20, height=20)
        canvas.create_oval(5, 5, 15, 15, fill=color)
        canvas.pack(side=tk.LEFT)

        label = tk.Label(frame, text=text)
        label.pack(side=tk.LEFT)

    def create_packet_sending_info(self):
        """
        Create or recreate the packet sending info in the info panel for the current set of users.
        """
        # Clear existing labels
        for label in self.packet_info_labels.values():
            label.destroy()
        self.packet_info_labels.clear()

        # Create new labels for current users
        for user in self.env.users:
            self.create_user_sending_info(user)

    def create_user_sending_info(self, user: User):
        """
        Create the packet sending info for a single user.

        Args:
            user (User): The user object containing the packet sending info.
        """
        user_name = user.id
        user_dest_name = user.dest.id if user.dest is not None else "None"
        user_offered_load = user.offeredLoad

        info_label = tk.Label(
            self.info_frame,
            text=f"{user_name} -> {user_dest_name}: {user_offered_load/1e6:.3f} Mb",
        )
        info_label.pack(anchor="w")
        self.packet_info_labels[user_name] = info_label

    def update_packet_sending_info(self):
        """
        Update the packet sending info in the info panel for the current set of users.
        """
        # Remove labels for users no longer present
        current_user_ids = {user.id for user in self.env.users}
        for user_id in list(self.packet_info_labels.keys()):
            if user_id not in current_user_ids:
                self.packet_info_labels[user_id].destroy()
                del self.packet_info_labels[user_id]

        # Create or update labels for current users
        for user in self.env.users:
            user: User
            user_name = user.id
            user_dest_name = user.dest.id if user.dest is not None else "None"
            user_offered_load = user.offeredLoad
            user_prev_data_del = user.prevDataDelivered

            if user_name not in self.packet_info_labels:
                self.create_user_sending_info(user)
            else:
                info_label = self.packet_info_labels[user_name]
                info_label.config(
                    text=f"{user_name} -> {user_dest_name}: {user_prev_data_del/1e6:.3f}/{user_offered_load/1e6:.3f} Mb"
                )

    def update_gui(self):
        """
        Continuously update the GUI based on the commands in the update queue.
        """
        if self.continue_updating:
            while not self.gui_update_queue.empty():
                cmd, *args = self.gui_update_queue.get()
                if cmd == "update":
                    self.update_entities_helper(*args)
                elif cmd == "reset":
                    self.resetComps(*args)
                elif cmd == "close":
                    self.on_closing()
            self.after(self.speed, self.update_gui)

    def create_entities(self, node_positions, jammer_positions, user_positions):
        """
        Create entities on the canvas.

        Args:
            node_positions (list): List of positions for the MANET nodes.
            jammer_positions (list): List of positions for the jammers.
            user_positions (list): List of positions for the users.
        """
        self.nodes = []
        self.node_texts = []
        self.jammers = []
        self.jammer_texts = []
        self.users = []
        self.user_texts = []
        self.edges = []

        self.create_components(node_positions, jammer_positions, user_positions)

    def create_components(self, node_positions, jammer_positions, user_positions):
        """
        Helper function to create entities on the canvas.

        Args:
            node_positions (list): List of positions for the MANET nodes.
            jammer_positions (list): List of positions for the jammers.
            user_positions (list): List of positions for the users.
        """
        # Create users
        for idx, user_pos in enumerate(user_positions):
            x, y = user_pos * self.scale
            self.users.append(
                self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill="green")
            )
            self.user_texts.append(
                self.canvas.create_text(
                    x,
                    y,
                    text=str(idx),
                    fill="white",
                    justify="center",
                    font=("Arial", 8),
                )
            )

        # Create nodes
        for idx, node_pos in enumerate(node_positions):
            x, y = node_pos * self.scale
            self.nodes.append(
                self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill="blue")
            )
            self.node_texts.append(
                self.canvas.create_text(
                    x,
                    y,
                    text=str(idx),
                    fill="white",
                    justify="center",
                    font=("Arial", 8),
                )
            )

        # Create jammers
        for idx, jammer_pos in enumerate(jammer_positions):
            x, y = jammer_pos * self.scale
            # Ensure jammer index is valid
            jammer = self.env.jammers[idx] if idx < len(self.env.jammers) else None
            fill_color = "red" if jammer and jammer.isActive else ""
            self.jammers.append(
                self.canvas.create_oval(
                    x - 5, y - 5, x + 5, y + 5, fill=fill_color, outline="red"
                )
            )
            self.jammer_texts.append(
                self.canvas.create_text(
                    x,
                    y,
                    text=str(idx),
                    fill="white",
                    justify="center",
                    font=("Arial", 8),
                )
            )

    def remove_components(self):
        """
        Remove all components (nodes, jammers, users, and their texts) from the canvas.
        """
        for item in (
            self.nodes
            + self.node_texts
            + self.jammers
            + self.jammer_texts
            + self.users
            + self.user_texts
            + self.edges
        ):
            self.canvas.delete(item)
        self.nodes.clear()
        self.node_texts.clear()
        self.jammers.clear()
        self.jammer_texts.clear()
        self.users.clear()
        self.user_texts.clear()
        self.edges.clear()

    def update_entities_helper(
        self, node_positions, jammer_positions, user_positions, edges, throughput
    ):
        """
        Helper function to update entities on the canvas.

        Args:
            node_positions (list): List of positions for the MANET nodes.
            jammer_positions (list): List of positions for the jammers.
            user_positions (list): List of positions for the users.
            edges (list): List of edges representing connections between entities.
            throughput (float): Current network throughput value.
        """
        # Check if component counts have changed
        if (
            len(node_positions) != len(self.nodes)
            or len(jammer_positions) != len(self.jammers)
            or len(user_positions) != len(self.users)
        ):
            self.remove_components()
            self.create_entities(node_positions, jammer_positions, user_positions)
        else:
            self.update_components(node_positions, jammer_positions, user_positions)
        self.update_edges(edges)
        self.update_throughput(throughput)
        self.update_packet_sending_info()

    def update_components(self, node_positions, jammer_positions, user_positions):
        """
        Update the positions of the components on the canvas.

        Args:
            node_positions (list): List of positions for the MANET nodes.
            jammer_positions (list): List of positions for the jammers.
            user_positions (list): List of positions for the users.
        """
        # Update nodes
        for i, node_pos in enumerate(node_positions):
            if i < len(self.nodes):
                x, y = node_pos * self.scale
                self.canvas.coords(self.nodes[i], x - 5, y - 5, x + 5, y + 5)
                self.canvas.coords(self.node_texts[i], x, y)

        # Update jammers
        for i, jammer_pos in enumerate(jammer_positions):
            if i < len(self.jammers):
                x, y = jammer_pos * self.scale
                jammer = self.env.jammers[i] if i < len(self.env.jammers) else None
                fill_color = "red" if jammer and jammer.isActive else ""
                self.canvas.coords(self.jammers[i], x - 5, y - 5, x + 5, y + 5)
                self.canvas.itemconfig(self.jammers[i], fill=fill_color)
                self.canvas.coords(self.jammer_texts[i], x, y)

        # Update users
        for i, user_pos in enumerate(user_positions):
            if i < len(self.users):
                x, y = user_pos * self.scale
                self.canvas.coords(self.users[i], x - 5, y - 5, x + 5, y + 5)
                self.canvas.coords(self.user_texts[i], x, y)

    def update_edges(self, edges):
        """
        Update the edges (connections) on the canvas.

        Args:
            edges (list): List of edges representing connections between entities.
        """
        for edge in self.edges:
            self.canvas.delete(edge)
        self.edges = []

        for u, v in edges:
            x1, y1 = u.pos * self.scale
            x2, y2 = v.pos * self.scale
            self.edges.append(
                self.canvas.create_line(x1, y1, x2, y2, fill="black", arrow=tk.LAST)
            )

    def update_throughput(self, throughput):
        """
        Update the network throughput label with the current value.
        Also display ep_step, updating only every N steps.
        Args:
            throughput (float): Current network throughput value.
        """
        ep_step = getattr(self.env, "ep_step", None)
        if ep_step is not None:
            # Only update every N steps
            if ep_step // self.ep_step_update_interval != self.last_ep_step_update:
                self.last_ep_step_update = ep_step // self.ep_step_update_interval
                throughput = round(throughput / 1e6, 2)
                self.throughput_label.config(
                    text=f"Network Throughput: {throughput:.3f} Mb | Step: {ep_step}"
                )
        else:
            throughput = round(throughput / 1e6, 2)
            self.throughput_label.config(
                text=f"Network Throughput: {throughput:.3f} Mb"
            )

    def create_speed_buttons(self):
        """
        Create buttons to control the speed of the simulation.
        """
        button_frame = tk.Frame(self)
        button_frame.pack(side=tk.BOTTOM, pady=10)

        slow_button = tk.Button(
            button_frame, text="Slow", command=lambda: self.changeSpeed(1 / 200)
        )
        slow_button.pack(side=tk.LEFT, padx=5)

        medium_button = tk.Button(
            button_frame, text="Medium", command=lambda: self.changeSpeed(1 / 50)
        )
        medium_button.pack(side=tk.LEFT, padx=5)

        fast_button = tk.Button(
            button_frame, text="Fast", command=lambda: self.changeSpeed(1)
        )
        fast_button.pack(side=tk.LEFT, padx=5)

        self.pause_button = tk.Button(
            button_frame, text="Pause", command=self.toggle_pause
        )
        self.pause_button.pack(side=tk.LEFT, padx=5)

    def toggle_pause(self):
        """
        Toggle the pause state of the simulation.
        """
        self.continue_updating = not self.continue_updating
        if self.continue_updating:
            self.pause_button.config(text="Pause")
            self.update_gui()
        else:
            self.pause_button.config(text="Unpause")

    def on_closing(self):
        """
        Handle the window close event.
        """
        self.continue_updating = False
        self.gui_state = "closed"
        self.quit()

    def on_minimize(self, event):
        """
        Handle the window minimize event.

        Args:
            event (tk.Event): The event object containing event data.
        """
        super().iconify()
        self.env.render_mode = None
        self.continue_updating = False

    def on_enlarge(self, event):
        """
        Handle the window enlarge (restore) event.

        Args:
            event (tk.Event): The event object containing event data.
        """
        super().deiconify()
        self.env.render_mode = "plot"
        self.continue_updating = True
        self.update_gui()
