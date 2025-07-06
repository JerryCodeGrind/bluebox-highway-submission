import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.animation as animation
import numpy as np
import threading
import time
import logging
from datetime import datetime, timedelta
from collections import deque
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalDisplayUI:
    def __init__(self, master=None, fullscreen=True):
        """
        Initialize medical device display UI
        
        Args:
            master: Tkinter master window (None for standalone)
            fullscreen: Whether to run in fullscreen mode
        """
        if master is None:
            self.root = tk.Tk()
        else:
            self.root = master
        
        # Configure main window
        self.root.title("Medical Device Monitor")
        self.root.configure(bg='#2c3e50')
        
        if fullscreen:
            self.root.attributes('-fullscreen', True)
            # For Raspberry Pi 7" display
            self.screen_width = 800
            self.screen_height = 480
        else:
            self.screen_width = 1024
            self.screen_height = 600
            self.root.geometry(f"{self.screen_width}x{self.screen_height}")
        
        # Bind escape key to exit fullscreen
        self.root.bind('<Escape>', self.toggle_fullscreen)
        self.root.bind('<F11>', self.toggle_fullscreen)
        
        # Initialize data storage
        self.vital_signs = {
            'heart_rate': deque(maxlen=300),  # 5 minutes at 1Hz
            'blood_pressure_sys': deque(maxlen=60),  # 1 hour at 1/min
            'blood_pressure_dia': deque(maxlen=60),
            'temperature': deque(maxlen=180),  # 3 hours at 1/min
            'spo2': deque(maxlen=300),
            'respiratory_rate': deque(maxlen=300)
        }
        
        self.timestamps = {
            'heart_rate': deque(maxlen=300),
            'blood_pressure': deque(maxlen=60),
            'temperature': deque(maxlen=180),
            'spo2': deque(maxlen=300),
            'respiratory_rate': deque(maxlen=300)
        }
        
        # Current values
        self.current_values = {
            'heart_rate': 0,
            'blood_pressure': '0/0',
            'temperature': 0.0,
            'spo2': 0,
            'respiratory_rate': 0,
            'status': 'Ready'
        }
        
        # UI state
        self.current_page = 'main'
        self.measurement_in_progress = False
        self.last_update = time.time()
        
        # Animation and threading
        self.animation_running = False
        self.update_thread = None
        self.data_lock = threading.Lock()
        
        # Create UI
        self.setup_styles()
        self.create_main_interface()
        self.start_animation()
        
        logger.info("Display UI initialized")
    
    def setup_styles(self):
        """Configure UI styles for medical device appearance"""
        self.style = ttk.Style()
        
        # Configure colors
        self.colors = {
            'bg_primary': '#2c3e50',    # Dark blue-gray
            'bg_secondary': '#34495e',   # Lighter blue-gray
            'accent': '#3498db',         # Blue
            'success': '#27ae60',        # Green
            'warning': '#f39c12',        # Orange
            'danger': '#e74c3c',         # Red
            'text_primary': '#ecf0f1',   # Light gray
            'text_secondary': '#bdc3c7'  # Medium gray
        }
        
        # Configure styles
        self.style.theme_use('clam')
        
        # Button styles
        self.style.configure('Medical.TButton',
                           background=self.colors['accent'],
                           foreground='white',
                           borderwidth=0,
                           focuscolor='none',
                           font=('Arial', 12, 'bold'))
        
        self.style.map('Medical.TButton',
                      background=[('active', '#2980b9')])
        
        # Frame styles
        self.style.configure('Medical.TFrame',
                           background=self.colors['bg_secondary'],
                           relief='raised',
                           borderwidth=2)
        
        # Label styles
        self.style.configure('Title.TLabel',
                           background=self.colors['bg_primary'],
                           foreground=self.colors['text_primary'],
                           font=('Arial', 16, 'bold'))
        
        self.style.configure('Value.TLabel',
                           background=self.colors['bg_secondary'],
                           foreground=self.colors['text_primary'],
                           font=('Arial', 24, 'bold'))
        
        self.style.configure('Unit.TLabel',
                           background=self.colors['bg_secondary'],
                           foreground=self.colors['text_secondary'],
                           font=('Arial', 12))
        
        self.style.configure('Status.TLabel',
                           background=self.colors['bg_primary'],
                           foreground=self.colors['text_primary'],
                           font=('Arial', 14))
    
    def create_main_interface(self):
        """Create the main medical device interface"""
        # Main container
        self.main_frame = tk.Frame(self.root, bg=self.colors['bg_primary'])
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title bar
        self.create_title_bar()
        
        # Main content area
        self.content_frame = tk.Frame(self.main_frame, bg=self.colors['bg_primary'])
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create pages
        self.create_dashboard_page()
        self.create_trends_page()
        self.create_settings_page()
        
        # Navigation bar
        self.create_navigation_bar()
        
        # Show dashboard by default
        self.show_page('dashboard')
    
    def create_title_bar(self):
        """Create title bar with status and time"""
        title_frame = tk.Frame(self.main_frame, bg=self.colors['bg_primary'], height=60)
        title_frame.pack(fill=tk.X, padx=10, pady=5)
        title_frame.pack_propagate(False)
        
        # Device title
        title_label = tk.Label(title_frame, 
                              text="Medical Device Monitor",
                              bg=self.colors['bg_primary'],
                              fg=self.colors['text_primary'],
                              font=('Arial', 18, 'bold'))
        title_label.pack(side=tk.LEFT, pady=15)
        
        # Status and time
        self.status_time_frame = tk.Frame(title_frame, bg=self.colors['bg_primary'])
        self.status_time_frame.pack(side=tk.RIGHT, pady=10)
        
        self.status_label = tk.Label(self.status_time_frame,
                                    text="Status: Ready",
                                    bg=self.colors['bg_primary'],
                                    fg=self.colors['success'],
                                    font=('Arial', 12))
        self.status_label.pack(side=tk.TOP)
        
        self.time_label = tk.Label(self.status_time_frame,
                                  text="",
                                  bg=self.colors['bg_primary'],
                                  fg=self.colors['text_secondary'],
                                  font=('Arial', 10))
        self.time_label.pack(side=tk.TOP)
    
    def create_dashboard_page(self):
        """Create main dashboard with vital signs"""
        self.dashboard_frame = tk.Frame(self.content_frame, bg=self.colors['bg_primary'])
        
        # Vital signs grid (2x3)
        self.vital_signs_frame = tk.Frame(self.dashboard_frame, bg=self.colors['bg_primary'])
        self.vital_signs_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Configure grid
        for i in range(2):
            self.vital_signs_frame.grid_rowconfigure(i, weight=1)
        for i in range(3):
            self.vital_signs_frame.grid_columnconfigure(i, weight=1)
        
        # Create vital sign displays
        self.vital_displays = {}
        
        # Heart Rate
        self.vital_displays['heart_rate'] = self.create_vital_display(
            self.vital_signs_frame, "Heart Rate", "0", "BPM", 0, 0, self.colors['danger'])
        
        # Blood Pressure
        self.vital_displays['blood_pressure'] = self.create_vital_display(
            self.vital_signs_frame, "Blood Pressure", "0/0", "mmHg", 0, 1, self.colors['warning'])
        
        # Temperature
        self.vital_displays['temperature'] = self.create_vital_display(
            self.vital_signs_frame, "Temperature", "0.0", "°C", 0, 2, self.colors['accent'])
        
        # SpO2
        self.vital_displays['spo2'] = self.create_vital_display(
            self.vital_signs_frame, "SpO₂", "0", "%", 1, 0, self.colors['success'])
        
        # Respiratory Rate
        self.vital_displays['respiratory_rate'] = self.create_vital_display(
            self.vital_signs_frame, "Respiratory Rate", "0", "BPM", 1, 1, self.colors['accent'])
        
        # Control buttons
        self.control_frame = tk.Frame(self.dashboard_frame, bg=self.colors['bg_primary'])
        self.control_frame.pack(fill=tk.X, padx=5, pady=10)
        
        # Measurement buttons
        self.start_button = tk.Button(self.control_frame,
                                     text="Start Measurement",
                                     bg=self.colors['success'],
                                     fg='white',
                                     font=('Arial', 14, 'bold'),
                                     command=self.start_measurement,
                                     relief='raised',
                                     bd=3)
        self.start_button.pack(side=tk.LEFT, padx=10, pady=5, ipadx=20, ipady=10)
        
        self.stop_button = tk.Button(self.control_frame,
                                    text="Stop Measurement",
                                    bg=self.colors['danger'],
                                    fg='white',
                                    font=('Arial', 14, 'bold'),
                                    command=self.stop_measurement,
                                    relief='raised',
                                    bd=3,
                                    state='disabled')
        self.stop_button.pack(side=tk.LEFT, padx=10, pady=5, ipadx=20, ipady=10)
        
        # Emergency button
        self.emergency_button = tk.Button(self.control_frame,
                                         text="EMERGENCY",
                                         bg='#c0392b',
                                         fg='white',
                                         font=('Arial', 16, 'bold'),
                                         command=self.emergency_action,
                                         relief='raised',
                                         bd=5)
        self.emergency_button.pack(side=tk.RIGHT, padx=10, pady=5, ipadx=30, ipady=15)
    
    def create_vital_display(self, parent, title, value, unit, row, col, color):
        """Create a vital sign display widget"""
        frame = tk.Frame(parent, bg=self.colors['bg_secondary'], relief='raised', bd=2)
        frame.grid(row=row, column=col, padx=5, pady=5, sticky='nsew')
        frame.grid_propagate(False)
        
        # Title
        title_label = tk.Label(frame, text=title,
                              bg=self.colors['bg_secondary'],
                              fg=self.colors['text_secondary'],
                              font=('Arial', 12, 'bold'))
        title_label.pack(pady=(10, 5))
        
        # Value
        value_label = tk.Label(frame, text=value,
                              bg=self.colors['bg_secondary'],
                              fg=color,
                              font=('Arial', 28, 'bold'))
        value_label.pack(pady=5)
        
        # Unit
        unit_label = tk.Label(frame, text=unit,
                             bg=self.colors['bg_secondary'],
                             fg=self.colors['text_secondary'],
                             font=('Arial', 10))
        unit_label.pack(pady=(0, 10))
        
        return {
            'frame': frame,
            'title': title_label,
            'value': value_label,
            'unit': unit_label
        }
    
    def create_trends_page(self):
        """Create trends page with charts"""
        self.trends_frame = tk.Frame(self.content_frame, bg=self.colors['bg_primary'])
        
        # Create matplotlib figure for trends
        self.fig = Figure(figsize=(10, 6), facecolor=self.colors['bg_primary'])
        self.fig.suptitle('Vital Signs Trends', color=self.colors['text_primary'], fontsize=16)
        
        # Create subplots
        self.ax1 = self.fig.add_subplot(2, 2, 1, facecolor=self.colors['bg_secondary'])
        self.ax2 = self.fig.add_subplot(2, 2, 2, facecolor=self.colors['bg_secondary'])
        self.ax3 = self.fig.add_subplot(2, 2, 3, facecolor=self.colors['bg_secondary'])
        self.ax4 = self.fig.add_subplot(2, 2, 4, facecolor=self.colors['bg_secondary'])
        
        # Configure subplot appearance
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.tick_params(colors=self.colors['text_secondary'])
            ax.spines['bottom'].set_color(self.colors['text_secondary'])
            ax.spines['top'].set_color(self.colors['text_secondary'])
            ax.spines['right'].set_color(self.colors['text_secondary'])
            ax.spines['left'].set_color(self.colors['text_secondary'])
        
        # Embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, self.trends_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Trend control buttons
        trend_control_frame = tk.Frame(self.trends_frame, bg=self.colors['bg_primary'])
        trend_control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Time range buttons
        for time_range, label in [('1h', '1 Hour'), ('6h', '6 Hours'), ('24h', '24 Hours')]:
            btn = tk.Button(trend_control_frame,
                           text=label,
                           bg=self.colors['accent'],
                           fg='white',
                           font=('Arial', 10),
                           command=lambda tr=time_range: self.update_trend_range(tr))
            btn.pack(side=tk.LEFT, padx=5, pady=5)
    
    def create_settings_page(self):
        """Create settings page"""
        self.settings_frame = tk.Frame(self.content_frame, bg=self.colors['bg_primary'])
        
        # Settings title
        settings_title = tk.Label(self.settings_frame,
                                 text="Device Settings",
                                 bg=self.colors['bg_primary'],
                                 fg=self.colors['text_primary'],
                                 font=('Arial', 18, 'bold'))
        settings_title.pack(pady=20)
        
        # Settings grid
        settings_grid = tk.Frame(self.settings_frame, bg=self.colors['bg_primary'])
        settings_grid.pack(fill=tk.BOTH, expand=True, padx=50)
        
        # Alarm thresholds
        self.create_setting_section(settings_grid, "Alarm Thresholds", 0)
        self.create_setting_section(settings_grid, "Display Options", 1)
        self.create_setting_section(settings_grid, "System Settings", 2)
    
    def create_setting_section(self, parent, title, row):
        """Create a settings section"""
        section_frame = tk.Frame(parent, bg=self.colors['bg_secondary'], relief='raised', bd=2)
        section_frame.grid(row=row, column=0, padx=10, pady=10, sticky='ew')
        
        # Section title
        title_label = tk.Label(section_frame, text=title,
                              bg=self.colors['bg_secondary'],
                              fg=self.colors['text_primary'],
                              font=('Arial', 14, 'bold'))
        title_label.pack(pady=10)
        
        # Add some example settings
        if title == "Alarm Thresholds":
            self.create_threshold_settings(section_frame)
        elif title == "Display Options":
            self.create_display_settings(section_frame)
        elif title == "System Settings":
            self.create_system_settings(section_frame)
    
    def create_threshold_settings(self, parent):
        """Create alarm threshold settings"""
        thresholds = [
            ("Heart Rate High", "120", "BPM"),
            ("Heart Rate Low", "60", "BPM"),
            ("Blood Pressure High", "140/90", "mmHg"),
            ("Temperature High", "37.5", "°C")
        ]
        
        for i, (label, default, unit) in enumerate(thresholds):
            row_frame = tk.Frame(parent, bg=self.colors['bg_secondary'])
            row_frame.pack(fill=tk.X, padx=10, pady=2)
            
            # Label
            tk.Label(row_frame, text=f"{label}:",
                    bg=self.colors['bg_secondary'],
                    fg=self.colors['text_primary'],
                    font=('Arial', 10)).pack(side=tk.LEFT)
            
            # Entry
            entry = tk.Entry(row_frame, width=10, font=('Arial', 10))
            entry.insert(0, default)
            entry.pack(side=tk.RIGHT, padx=(0, 5))
            
            # Unit
            tk.Label(row_frame, text=unit,
                    bg=self.colors['bg_secondary'],
                    fg=self.colors['text_secondary'],
                    font=('Arial', 10)).pack(side=tk.RIGHT)
    
    def create_display_settings(self, parent):
        """Create display settings"""
        # Brightness control
        brightness_frame = tk.Frame(parent, bg=self.colors['bg_secondary'])
        brightness_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(brightness_frame, text="Brightness:",
                bg=self.colors['bg_secondary'],
                fg=self.colors['text_primary'],
                font=('Arial', 10)).pack(side=tk.LEFT)
        
        self.brightness_scale = tk.Scale(brightness_frame, from_=10, to=100,
                                        orient=tk.HORIZONTAL,
                                        bg=self.colors['bg_secondary'],
                                        fg=self.colors['text_primary'],
                                        command=self.adjust_brightness)
        self.brightness_scale.set(80)
        self.brightness_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=10)
        
        # Auto-dim checkbox
        auto_dim_var = tk.BooleanVar()
        auto_dim_check = tk.Checkbutton(parent, text="Auto-dim at night",
                                       variable=auto_dim_var,
                                       bg=self.colors['bg_secondary'],
                                       fg=self.colors['text_primary'],
                                       selectcolor=self.colors['accent'])
        auto_dim_check.pack(padx=10, pady=5)
    
    def create_system_settings(self, parent):
        """Create system settings"""
        settings_list = [
            ("WiFi Setup", self.wifi_setup),
            ("Calibrate Sensors", self.calibrate_sensors),
            ("Export Data", self.export_data),
            ("System Info", self.show_system_info)
        ]
        
        for text, command in settings_list:
            btn = tk.Button(parent, text=text,
                           bg=self.colors['accent'],
                           fg='white',
                           font=('Arial', 10),
                           command=command,
                           width=20)
            btn.pack(pady=5)
    
    def create_navigation_bar(self):
        """Create bottom navigation bar"""
        nav_frame = tk.Frame(self.main_frame, bg=self.colors['bg_secondary'], height=60)
        nav_frame.pack(fill=tk.X, side=tk.BOTTOM)
        nav_frame.pack_propagate(False)
        
        # Navigation buttons
        nav_buttons = [
            ("Dashboard", "dashboard", "🏠"),
            ("Trends", "trends", "📊"),
            ("Settings", "settings", "⚙️")
        ]
        
        self.nav_buttons = {}
        for text, page, icon in nav_buttons:
            btn = tk.Button(nav_frame,
                           text=f"{icon}\n{text}",
                           bg=self.colors['bg_secondary'],
                           fg=self.colors['text_primary'],
                           font=('Arial', 10),
                           relief='flat',
                           command=lambda p=page: self.show_page(p),
                           width=12)
            btn.pack(side=tk.LEFT, fill=tk.Y, expand=True)
            self.nav_buttons[page] = btn
    
    def show_page(self, page_name):
        """Show specified page"""
        # Hide all pages
        for frame in [self.dashboard_frame, self.trends_frame, self.settings_frame]:
            frame.pack_forget()
        
        # Show selected page
        if page_name == 'dashboard':
            self.dashboard_frame.pack(fill=tk.BOTH, expand=True)
        elif page_name == 'trends':
            self.trends_frame.pack(fill=tk.BOTH, expand=True)
            self.update_trends()
        elif page_name == 'settings':
            self.settings_frame.pack(fill=tk.BOTH, expand=True)
        
        # Update navigation button states
        for btn_name, btn in self.nav_buttons.items():
            if btn_name == page_name:
                btn.configure(bg=self.colors['accent'])
            else:
                btn.configure(bg=self.colors['bg_secondary'])
        
        self.current_page = page_name
    
    def update_vital_display(self, vital_type, value, status='normal'):
        """Update a vital sign display"""
        if vital_type not in self.vital_displays:
            return
        
        display = self.vital_displays[vital_type]
        
        # Update value
        if isinstance(value, float):
            display['value'].config(text=f"{value:.1f}")
        else:
            display['value'].config(text=str(value))
        
        # Update color based on status
        color_map = {
            'normal': self.colors['success'],
            'warning': self.colors['warning'],
            'danger': self.colors['danger'],
            'offline': self.colors['text_secondary']
        }
        
        color = color_map.get(status, self.colors['text_primary'])
        display['value'].config(fg=color)
        
        # Store in history
        with self.data_lock:
            if vital_type in self.vital_signs:
                self.vital_signs[vital_type].append(value)
                timestamp = datetime.now()
                
                if vital_type == 'blood_pressure':
                    self.timestamps['blood_pressure'].append(timestamp)
                else:
                    self.timestamps[vital_type].append(timestamp)
    
    def start_measurement(self):
        """Start measurement process"""
        self.measurement_in_progress = True
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')
        
        self.update_status("Measuring...", self.colors['warning'])
        
        # Start measurement thread
        measurement_thread = threading.Thread(target=self.measurement_process, daemon=True)
        measurement_thread.start()
        
        logger.info("Measurement started")
    
    def stop_measurement(self):
        """Stop measurement process"""
        self.measurement_in_progress = False
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        
        self.update_status("Ready", self.colors['success'])
        logger.info("Measurement stopped")
    
    def measurement_process(self):
        """Simulated measurement process"""
        # This would integrate with your actual sensor modules
        measurement_steps = [
            ("Initializing sensors...", 2),
            ("Measuring heart rate...", 5),
            ("Taking blood pressure...", 8),
            ("Reading temperature...", 3),
            ("Measuring SpO2...", 4),
            ("Analyzing data...", 2)
        ]
        
        for step, duration in measurement_steps:
            if not self.measurement_in_progress:
                break
            
            self.root.after(0, lambda s=step: self.update_status(s, self.colors['warning']))
            
            # Simulate measurement with fake data
            for i in range(duration):
                if not self.measurement_in_progress:
                    break
                time.sleep(1)
                
                # Generate fake vital signs
                self.root.after(0, lambda: self.update_fake_vitals())
        
        if self.measurement_in_progress:
            self.root.after(0, lambda: self.update_status("Complete", self.colors['success']))
            time.sleep(2)
            self.root.after(0, self.stop_measurement)
    
    def update_fake_vitals(self):
        """Update with simulated vital signs data"""
        import random
        
        # Generate realistic fake data
        hr = random.randint(65, 85) + random.randint(-5, 5)
        bp_sys = random.randint(110, 130) + random.randint(-5, 5)
        bp_dia = random.randint(70, 85) + random.randint(-3, 3)
        temp = 36.5 + random.uniform(-0.5, 0.5)
        spo2 = random.randint(96, 100)
        rr = random.randint(12, 18) + random.randint(-2, 2)
        
        # Update displays
        self.update_vital_display('heart_rate', hr)
        self.update_vital_display('blood_pressure', f"{bp_sys}/{bp_dia}")
        self.update_vital_display('temperature', temp)
        self.update_vital_display('spo2', spo2)
        self.update_vital_display('respiratory_rate', rr)
        
        # Store individual BP values
        with self.data_lock:
            self.vital_signs['blood_pressure_sys'].append(bp_sys)
            self.vital_signs['blood_pressure_dia'].append(bp_dia)
    
    def update_trends(self):
        """Update trend charts"""
        with self.data_lock:
            # Clear all subplots
            self.ax1.clear()
            self.ax2.clear()
            self.ax3.clear()
            self.ax4.clear()
            
            # Heart Rate
            if self.vital_signs['heart_rate'] and self.timestamps['heart_rate']:
                times = list(self.timestamps['heart_rate'])
                values = list(self.vital_signs['heart_rate'])
                
                if times and values:
                    self.ax1.plot(times, values, color='red', linewidth=2)
                    self.ax1.set_title('Heart Rate', color=self.colors['text_primary'])
                    self.ax1.set_ylabel('BPM', color=self.colors['text_primary'])
            
            # Blood Pressure
            if (self.vital_signs['blood_pressure_sys'] and 
                self.vital_signs['blood_pressure_dia'] and 
                self.timestamps['blood_pressure']):
                
                times = list(self.timestamps['blood_pressure'])
                sys_values = list(self.vital_signs['blood_pressure_sys'])
                dia_values = list(self.vital_signs['blood_pressure_dia'])
                
                if times and sys_values and dia_values:
                    self.ax2.plot(times, sys_values, color='orange', linewidth=2, label='Systolic')
                    self.ax2.plot(times, dia_values, color='blue', linewidth=2, label='Diastolic')
                    self.ax2.set_title('Blood Pressure', color=self.colors['text_primary'])
                    self.ax2.set_ylabel('mmHg', color=self.colors['text_primary'])
                    self.ax2.legend()
            
            # Temperature
            if self.vital_signs['temperature'] and self.timestamps['temperature']:
                times = list(self.timestamps['temperature'])
                values = list(self.vital_signs['temperature'])
                
                if times and values:
                    self.ax3.plot(times, values, color='green', linewidth=2)
                    self.ax3.set_title('Temperature', color=self.colors['text_primary'])
                    self.ax3.set_ylabel('°C', color=self.colors['text_primary'])
            
            # SpO2
            if self.vital_signs['spo2'] and self.timestamps['spo2']:
                times = list(self.timestamps['spo2'])
                values = list(self.vital_signs['spo2'])
                
                if times and values:
                    self.ax4.plot(times, values, color='cyan', linewidth=2)
                    self.ax4.set_title('SpO₂', color=self.colors['text_primary'])
                    self.ax4.set_ylabel('%', color=self.colors['text_primary'])
            
            # Configure all axes
            for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
                ax.tick_params(colors=self.colors['text_secondary'])
                ax.grid(True, alpha=0.3)
                ax.set_facecolor(self.colors['bg_secondary'])
        
        self.canvas.draw()
    
    def update_status(self, status_text, color=None):
        """Update status display"""
        if color is None:
            color = self.colors['text_primary']
        
        self.status_label.config(text=f"Status: {status_text}", fg=color)
        self.current_values['status'] = status_text
    
    def update_time_display(self):
        """Update time display"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.time_label.config(text=current_time)
    
    def start_animation(self):
        """Start UI animation loop"""
        self.animation_running = True
        self.animate()
    
    def animate(self):
        """Animation loop for real-time updates"""
        if not self.animation_running:
            return
        
        # Update time
        self.update_time_display()
        
        # Update trends if on trends page
        if self.current_page == 'trends':
            # Update trends every 5 seconds
            if time.time() - self.last_update > 5:
                self.update_trends()
                self.last_update = time.time()
        
        # Schedule next update
        self.root.after(1000, self.animate)  # Update every second
    
    def emergency_action(self):
        """Handle emergency button press"""
        result = messagebox.askyesno("Emergency", 
                                   "Emergency button pressed!\n\nThis will:\n"
                                   "• Stop all measurements\n"
                                   "• Alert medical staff\n"
                                   "• Save current data\n\n"
                                   "Continue?")
        if result:
            self.stop_measurement()
            self.update_status("EMERGENCY ALERT", self.colors['danger'])
            # Add emergency procedures here
            logger.warning("Emergency button activated")
    
    def adjust_brightness(self, value):
        """Adjust display brightness"""
        # This would control actual display brightness on Raspberry Pi
        logger.info(f"Brightness set to {value}%")
    
    def wifi_setup(self):
        """WiFi setup dialog"""
        messagebox.showinfo("WiFi Setup", "WiFi configuration would open here")
    
    def calibrate_sensors(self):
        """Sensor calibration"""
        messagebox.showinfo("Calibration", "Sensor calibration would start here")
    
    def export_data(self):
        """Export measurement data"""
        messagebox.showinfo("Export", "Data export would start here")
    
    def show_system_info(self):
        """Show system information"""
        info = ("System Information\n\n"
                "Device: Medical Monitor v1.0\n"
                "OS: Raspberry Pi OS\n"
                "Memory: 4GB\n"
                "Storage: 32GB\n"
                "Network: Connected")
        messagebox.showinfo("System Info", info)
    
    def update_trend_range(self, time_range):
        """Update trend display time range"""
        logger.info(f"Trend range changed to {time_range}")
        self.update_trends()
    
    def toggle_fullscreen(self, event=None):
        """Toggle fullscreen mode"""
        current_state = self.root.attributes('-fullscreen')
        self.root.attributes('-fullscreen', not current_state)
    
    def cleanup(self):
        """Cleanup resources"""
        self.animation_running = False
        logger.info("Display UI cleaned up")
    
    def run(self):
        """Run the UI main loop"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.cleanup()

# Integration class for sensor data
class SensorDataInterface:
    """Interface for connecting real sensor data to the display"""
    
    def __init__(self, display_ui):
        self.display = display_ui
        self.sensors = {}
    
    def register_sensor(self, sensor_type, sensor_object):
        """Register a sensor object"""
        self.sensors[sensor_type] = sensor_object
    
    def start_data_collection(self):
        """Start collecting data from all sensors"""
        def collect_data():
            while True:
                try:
                    # Collect from each sensor type
                    if 'pulse_oximeter' in self.sensors:
                        data = self.sensors['pulse_oximeter'].get_current_data()
                        self.display.root.after(0, lambda: self.display.update_vital_display(
                            'heart_rate', data.get('heart_rate', 0)))
                        self.display.root.after(0, lambda: self.display.update_vital_display(
                            'spo2', data.get('spo2', 0)))
                    
                    if 'temperature' in self.sensors:
                        data = self.sensors['temperature'].get_current_data()
                        self.display.root.after(0, lambda: self.display.update_vital_display(
                            'temperature', data.get('temperature', 0.0)))
                    
                    if 'blood_pressure' in self.sensors:
                        data = self.sensors['blood_pressure'].get_last_measurement()
                        if data:
                            bp_value = f"{data.get('systolic', 0)}/{data.get('diastolic', 0)}"
                            self.display.root.after(0, lambda: self.display.update_vital_display(
                                'blood_pressure', bp_value))
                    
                    time.sleep(1)  # Update every second
                    
                except Exception as e:
                    logger.error(f"Data collection error: {e}")
                    time.sleep(1)
        
        # Start in background thread
        data_thread = threading.Thread(target=collect_data, daemon=True)
        data_thread.start()

# Example usage
if __name__ == "__main__":
    # Create and run the medical display UI
    app = MedicalDisplayUI(fullscreen=False)  # Set to True for Raspberry Pi
    
    try:
        app.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        app.cleanup()