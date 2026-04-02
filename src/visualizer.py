"""
Graphical Visualization Module for Content Generation System

Provides interactive GUI for:
- Visualizing zone descriptions and their relationships
- Displaying generated textures with detail detection overlays
- Showing integrity check results
- Real-time generation monitoring
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass
from pathlib import Path
import threading
import logging

# Try to import PIL for image display
try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logging.warning("PIL not available, image display will be limited")

logger = logging.getLogger(__name__)


@dataclass
class ZoneData:
    """Represents zone data for visualization."""
    zone_id: str
    description: str
    timestamp: Optional[str]
    consistency_score: float
    temporal_score: float
    is_valid: bool


@dataclass
class TextureData:
    """Represents texture data for visualization."""
    texture_id: str
    image_path: Optional[str]
    prompt: str
    required_details: List[str]
    detected_details: List[str]
    missing_details: List[str]
    is_valid: bool


class ContentGenerationVisualizer:
    """
    Main graphical interface for the content generation system.
    
    Provides tabs for:
    - Text zone generation and visualization
    - Texture generation and detail checking
    - Integrity status overview
    """
    
    def __init__(self, root: tk.Tk):
        """
        Initialize the visualizer.
        
        Args:
            root: Tkinter root window
        """
        self.root = root
        self.root.title("Генерация контента с контролем целостности")
        self.root.geometry("1200x800")
        
        # Data storage
        self.zones: List[ZoneData] = []
        self.textures: List[TextureData] = []
        self.current_zone_var = tk.StringVar()
        self.current_texture_var = tk.StringVar()
        
        # Setup UI
        self._setup_styles()
        self._create_menu()
        self._create_tabs()
        self._create_status_bar()
        
        logger.info("Visualizer initialized")
    
    def _setup_styles(self):
        """Configure ttk styles."""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        style.configure('Valid.TLabel', foreground='green')
        style.configure('Invalid.TLabel', foreground='red')
        style.configure('Warning.TLabel', foreground='orange')
        
        style.configure('Title.TLabel', font=('Helvetica', 16, 'bold'))
        style.configure('Subtitle.TLabel', font=('Helvetica', 12, 'bold'))
    
    def _create_menu(self):
        """Create menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Файл", menu=file_menu)
        file_menu.add_command(label="Сохранить проект", command=self._save_project)
        file_menu.add_command(label="Загрузить проект", command=self._load_project)
        file_menu.add_separator()
        file_menu.add_command(label="Выход", command=self.root.quit)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Помощь", menu=help_menu)
        help_menu.add_command(label="О программе", command=self._show_about)
    
    def _create_tabs(self):
        """Create main tabbed interface."""
        self.tabs = ttk.Notebook(self.root)
        self.tabs.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Tab 1: Text Zone Generation
        self.text_tab = ttk.Frame(self.tabs)
        self.tabs.add(self.text_tab, text="📝 Генерация текста")
        self._setup_text_tab()
        
        # Tab 2: Texture Generation
        self.texture_tab = ttk.Frame(self.tabs)
        self.tabs.add(self.texture_tab, text="🎨 Генерация текстур")
        self._setup_texture_tab()
        
        # Tab 3: Integrity Overview
        self.overview_tab = ttk.Frame(self.tabs)
        self.tabs.add(self.overview_tab, text="📊 Обзор целостности")
        self._setup_overview_tab()
    
    def _setup_text_tab(self):
        """Setup text generation tab."""
        # Left panel - Controls
        left_panel = ttk.LabelFrame(self.text_tab, text="Управление генерацией", padding=10)
        left_panel.pack(side='left', fill='y', padx=10, pady=10)
        
        # Zone ID input
        ttk.Label(left_panel, text="ID зоны:").pack(anchor='w', pady=(0, 5))
        self.zone_id_entry = ttk.Entry(left_panel, width=30)
        self.zone_id_entry.pack(fill='x', pady=(0, 10))
        self.zone_id_entry.insert(0, "zone_001")
        
        # Timestamp input
        ttk.Label(left_panel, text="Временная метка:").pack(anchor='w', pady=(0, 5))
        self.timestamp_entry = ttk.Entry(left_panel, width=30)
        self.timestamp_entry.pack(fill='x', pady=(0, 10))
        self.timestamp_entry.insert(0, "2024-01-01 10:00:00")
        
        # Prompt input
        ttk.Label(left_panel, text="Промпт:").pack(anchor='w', pady=(0, 5))
        self.prompt_text = scrolledtext.ScrolledText(left_panel, width=40, height=8)
        self.prompt_text.pack(fill='x', pady=(0, 10))
        self.prompt_text.insert('1.0', "Опишите входную зону здания")
        
        # Generate button
        self.generate_btn = ttk.Button(
            left_panel,
            text="🚀 Сгенерировать",
            command=self._generate_zone
        )
        self.generate_btn.pack(fill='x', pady=5)
        
        # Check integrity button
        self.check_btn = ttk.Button(
            left_panel,
            text="✓ Проверить целостность",
            command=self._check_zone_integrity
        )
        self.check_btn.pack(fill='x', pady=5)
        
        # Clear history button
        ttk.Button(
            left_panel,
            text="🗑️ Очистить историю",
            command=self._clear_zone_history
        ).pack(fill='x', pady=5)
        
        # Right panel - Results
        right_panel = ttk.LabelFrame(self.text_tab, text="Результаты", padding=10)
        right_panel.pack(side='right', fill='both', expand=True, padx=10, pady=10)
        
        # Zone list
        ttk.Label(right_panel, text="Список зон:").pack(anchor='w', pady=(0, 5))
        self.zone_listbox = tk.Listbox(right_panel, height=10, width=50)
        self.zone_listbox.pack(fill='x', pady=(0, 10))
        self.zone_listbox.bind('<<ListboxSelect>>', self._on_zone_select)
        
        # Zone details
        ttk.Label(right_panel, text="Описание зоны:").pack(anchor='w', pady=(0, 5))
        self.zone_description = scrolledtext.ScrolledText(right_panel, height=15, width=50)
        self.zone_description.pack(fill='both', expand=True)
        
        # Integrity status
        self.integrity_frame = ttk.LabelFrame(right_panel, text="Статус целостности", padding=10)
        self.integrity_frame.pack(fill='x', pady=10)
        
        self.consistency_label = ttk.Label(self.integrity_frame, text="Согласованность: N/A")
        self.consistency_label.pack(anchor='w')
        
        self.temporal_label = ttk.Label(self.integrity_frame, text="Временная логика: N/A")
        self.temporal_label.pack(anchor='w')
        
        self.validity_label = ttk.Label(self.integrity_frame, text="Статус: N/A")
        self.validity_label.pack(anchor='w')
    
    def _setup_texture_tab(self):
        """Setup texture generation tab."""
        # Left panel - Controls
        left_panel = ttk.LabelFrame(self.texture_tab, text="Управление генерацией", padding=10)
        left_panel.pack(side='left', fill='y', padx=10, pady=10)
        
        # Texture ID input
        ttk.Label(left_panel, text="ID текстуры:").pack(anchor='w', pady=(0, 5))
        self.texture_id_entry = ttk.Entry(left_panel, width=30)
        self.texture_id_entry.pack(fill='x', pady=(0, 10))
        self.texture_id_entry.insert(0, "texture_001")
        
        # Prompt input
        ttk.Label(left_panel, text="Промпт:").pack(anchor='w', pady=(0, 5))
        self.texture_prompt = scrolledtext.ScrolledText(left_panel, width=40, height=6)
        self.texture_prompt.pack(fill='x', pady=(0, 10))
        self.texture_prompt.insert('1.0', "Текстура деревянной двери с ручкой")
        
        # Required details
        ttk.Label(left_panel, text="Обязательные детали (через запятую):").pack(anchor='w', pady=(0, 5))
        self.details_entry = ttk.Entry(left_panel, width=30)
        self.details_entry.pack(fill='x', pady=(0, 10))
        self.details_entry.insert(0, "ручка двери, выключатель")
        
        # Generate button
        ttk.Button(
            left_panel,
            text="🚀 Сгенерировать текстуру",
            command=self._generate_texture
        ).pack(fill='x', pady=5)
        
        # Check details button
        ttk.Button(
            left_panel,
            text="✓ Проверить детали",
            command=self._check_texture_details
        ).pack(fill='x', pady=5)
        
        # Load image button
        ttk.Button(
            left_panel,
            text="📂 Загрузить изображение",
            command=self._load_texture_image
        ).pack(fill='x', pady=5)
        
        # Right panel - Results
        right_panel = ttk.LabelFrame(self.texture_tab, text="Результаты", padding=10)
        right_panel.pack(side='right', fill='both', expand=True, padx=10, pady=10)
        
        # Texture list
        ttk.Label(right_panel, text="Список текстур:").pack(anchor='w', pady=(0, 5))
        self.texture_listbox = tk.Listbox(right_panel, height=8, width=50)
        self.texture_listbox.pack(fill='x', pady=(0, 10))
        self.texture_listbox.bind('<<ListboxSelect>>', self._on_texture_select)
        
        # Image display area
        self.image_frame = ttk.LabelFrame(right_panel, text="Изображение", padding=10)
        self.image_frame.pack(fill='both', expand=True, pady=(0, 10))
        
        self.image_label = ttk.Label(self.image_frame, text="Нет изображения")
        self.image_label.pack(expand=True)
        
        # Details status
        self.details_frame = ttk.LabelFrame(right_panel, text="Статус деталей", padding=10)
        self.details_frame.pack(fill='x')
        
        self.detected_label = ttk.Label(self.details_frame, text="Найдено: N/A")
        self.detected_label.pack(anchor='w')
        
        self.missing_label = ttk.Label(self.details_frame, text="Отсутствует: N/A")
        self.missing_label.pack(anchor='w')
        
        self.texture_validity_label = ttk.Label(self.details_frame, text="Статус: N/A")
        self.texture_validity_label.pack(anchor='w')
    
    def _setup_overview_tab(self):
        """Setup integrity overview tab."""
        # Summary frame
        summary_frame = ttk.LabelFrame(self.overview_tab, text="Общая статистика", padding=10)
        summary_frame.pack(fill='x', padx=10, pady=10)
        
        self.total_zones_label = ttk.Label(summary_frame, text="Всего зон: 0")
        self.total_zones_label.pack(side='left', padx=20)
        
        self.valid_zones_label = ttk.Label(summary_frame, text="Валидные зоны: 0")
        self.valid_zones_label.pack(side='left', padx=20)
        
        self.total_textures_label = ttk.Label(summary_frame, text="Всего текстур: 0")
        self.total_textures_label.pack(side='left', padx=20)
        
        self.valid_textures_label = ttk.Label(summary_frame, text="Валидные текстуры: 0")
        self.valid_textures_label.pack(side='left', padx=20)
        
        # Zones table
        zones_frame = ttk.LabelFrame(self.overview_tab, text="Зоны", padding=10)
        zones_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        columns = ('ID', 'Время', 'Согласованность', 'Временная логика', 'Статус')
        self.zones_tree = ttk.Treeview(zones_frame, columns=columns, show='headings', height=10)
        
        for col in columns:
            self.zones_tree.heading(col, text=col)
            self.zones_tree.column(col, width=150)
        
        self.zones_tree.pack(side='left', fill='both', expand=True)
        
        zones_scrollbar = ttk.Scrollbar(zones_frame, orient='vertical', command=self.zones_tree.yview)
        zones_scrollbar.pack(side='right', fill='y')
        self.zones_tree.configure(yscrollcommand=zones_scrollbar.set)
        
        # Textures table
        textures_frame = ttk.LabelFrame(self.overview_tab, text="Текстуры", padding=10)
        textures_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        tex_columns = ('ID', 'Детали найдены', 'Детали отсутствуют', 'Статус')
        self.textures_tree = ttk.Treeview(textures_frame, columns=tex_columns, show='headings', height=8)
        
        for col in tex_columns:
            self.textures_tree.heading(col, text=col)
            self.textures_tree.column(col, width=200)
        
        self.textures_tree.pack(side='left', fill='both', expand=True)
        
        textures_scrollbar = ttk.Scrollbar(textures_frame, orient='vertical', command=self.textures_tree.yview)
        textures_scrollbar.pack(side='right', fill='y')
        self.textures_tree.configure(yscrollcommand=textures_scrollbar.set)
        
        # Refresh button
        ttk.Button(
            self.overview_tab,
            text="🔄 Обновить",
            command=self._refresh_overview
        ).pack(pady=10)
    
    def _create_status_bar(self):
        """Create status bar at bottom."""
        self.status_bar = ttk.Label(
            self.root,
            text="Готов",
            relief='sunken',
            anchor='w'
        )
        self.status_bar.pack(side='bottom', fill='x')
    
    # Text generation methods
    def _generate_zone(self):
        """Handle zone generation request."""
        zone_id = self.zone_id_entry.get()
        timestamp = self.timestamp_entry.get()
        prompt = self.prompt_text.get('1.0', tk.END).strip()
        
        if not zone_id or not prompt:
            messagebox.showwarning("Предупреждение", "ID зоны и промпт обязательны")
            return
        
        self.status_bar.config(text=f"Генерация зоны {zone_id}...")
        
        # Simulate generation (in real app, call the generator)
        self._simulate_zone_generation(zone_id, prompt, timestamp)
    
    def _simulate_zone_generation(self, zone_id: str, prompt: str, timestamp: str):
        """Simulate zone generation for demo."""
        # Mock generation result
        description = f"[Сгенерировано] {prompt} - зона {zone_id}"
        
        zone_data = ZoneData(
            zone_id=zone_id,
            description=description,
            timestamp=timestamp,
            consistency_score=0.95,
            temporal_score=0.90,
            is_valid=True
        )
        
        self.zones.append(zone_data)
        self._update_zone_list()
        self._select_zone(zone_id)
        
        self.status_bar.config(text=f"Зона {zone_id} сгенерирована")
        messagebox.showinfo("Успех", f"Зона {zone_id} успешно сгенерирована")
    
    def _check_zone_integrity(self):
        """Check integrity of selected zone."""
        selection = self.zone_listbox.curselection()
        if not selection:
            messagebox.showwarning("Предупреждение", "Выберите зону для проверки")
            return
        
        index = selection[0]
        zone = self.zones[index]
        
        # Update integrity display
        self.consistency_label.config(
            text=f"Согласованность: {zone.consistency_score:.2f}",
            style='Valid.TLabel' if zone.consistency_score > 0.7 else 'Invalid.TLabel'
        )
        
        self.temporal_label.config(
            text=f"Временная логика: {zone.temporal_score:.2f}",
            style='Valid.TLabel' if zone.temporal_score > 0.7 else 'Invalid.TLabel'
        )
        
        self.validity_label.config(
            text=f"Статус: {'✓ Валидна' if zone.is_valid else '✗ Невалидна'}",
            style='Valid.TLabel' if zone.is_valid else 'Invalid.TLabel'
        )
        
        self.status_bar.config(text=f"Целостность зоны {zone.zone_id} проверена")
    
    def _on_zone_select(self, event):
        """Handle zone selection."""
        selection = self.zone_listbox.curselection()
        if selection:
            index = selection[0]
            zone = self.zones[index]
            self._display_zone_details(zone)
    
    def _display_zone_details(self, zone: ZoneData):
        """Display zone details in the text area."""
        self.zone_description.delete('1.0', tk.END)
        self.zone_description.insert('1.0', zone.description)
        
        # Update integrity labels
        self.consistency_label.config(
            text=f"Согласованность: {zone.consistency_score:.2f}",
            style='Valid.TLabel' if zone.consistency_score > 0.7 else 'Invalid.TLabel'
        )
        
        self.temporal_label.config(
            text=f"Временная логика: {zone.temporal_score:.2f}",
            style='Valid.TLabel' if zone.temporal_score > 0.7 else 'Invalid.TLabel'
        )
        
        self.validity_label.config(
            text=f"Статус: {'✓ Валидна' if zone.is_valid else '✗ Невалидна'}",
            style='Valid.TLabel' if zone.is_valid else 'Invalid.TLabel'
        )
    
    def _update_zone_list(self):
        """Update the zone listbox."""
        self.zone_listbox.delete(0, tk.END)
        for zone in self.zones:
            status = "✓" if zone.is_valid else "✗"
            self.zone_listbox.insert(tk.END, f"{status} {zone.zone_id}")
    
    def _select_zone(self, zone_id: str):
        """Select a zone by ID."""
        for i, zone in enumerate(self.zones):
            if zone.zone_id == zone_id:
                self.zone_listbox.selection_clear(0, tk.END)
                self.zone_listbox.selection_set(i)
                self._display_zone_details(zone)
                break
    
    def _clear_zone_history(self):
        """Clear zone history."""
        if messagebox.askyesno("Подтверждение", "Очистить историю зон?"):
            self.zones.clear()
            self._update_zone_list()
            self.zone_description.delete('1.0', tk.END)
            self.consistency_label.config(text="Согласованность: N/A")
            self.temporal_label.config(text="Временная логика: N/A")
            self.validity_label.config(text="Статус: N/A")
            self.status_bar.config(text="История зон очищена")
    
    # Texture generation methods
    def _generate_texture(self):
        """Handle texture generation request."""
        texture_id = self.texture_id_entry.get()
        prompt = self.texture_prompt.get('1.0', tk.END).strip()
        details = [d.strip() for d in self.details_entry.get().split(',') if d.strip()]
        
        if not texture_id or not prompt:
            messagebox.showwarning("Предупреждение", "ID текстуры и промпт обязательны")
            return
        
        self.status_bar.config(text=f"Генерация текстуры {texture_id}...")
        
        # Simulate generation
        self._simulate_texture_generation(texture_id, prompt, details)
    
    def _simulate_texture_generation(self, texture_id: str, prompt: str, required_details: List[str]):
        """Simulate texture generation for demo."""
        # Mock generation result
        texture_data = TextureData(
            texture_id=texture_id,
            image_path=None,
            prompt=prompt,
            required_details=required_details,
            detected_details=required_details[:len(required_details)//2 + 1] if required_details else [],
            missing_details=required_details[len(required_details)//2 + 1:] if required_details else [],
            is_valid=True
        )
        
        self.textures.append(texture_data)
        self._update_texture_list()
        self._select_texture(texture_id)
        
        self.status_bar.config(text=f"Текстура {texture_id} сгенерирована")
        messagebox.showinfo("Успех", f"Текстура {texture_id} успешно сгенерирована")
    
    def _check_texture_details(self):
        """Check details of selected texture."""
        selection = self.texture_listbox.curselection()
        if not selection:
            messagebox.showwarning("Предупреждение", "Выберите текстуру для проверки")
            return
        
        index = selection[0]
        texture = self.textures[index]
        
        # Update details display
        detected = ', '.join(texture.detected_details) if texture.detected_details else 'Нет'
        missing = ', '.join(texture.missing_details) if texture.missing_details else 'Нет'
        
        self.detected_label.config(
            text=f"Найдено: {detected}",
            style='Valid.TLabel'
        )
        
        self.missing_label.config(
            text=f"Отсутствует: {missing}",
            style='Invalid.TLabel' if texture.missing_details else 'Valid.TLabel'
        )
        
        self.texture_validity_label.config(
            text=f"Статус: {'✓ Все детали на месте' if texture.is_valid else '✗ Отсутствуют детали'}",
            style='Valid.TLabel' if texture.is_valid else 'Invalid.TLabel'
        )
        
        self.status_bar.config(text=f"Детали текстуры {texture.texture_id} проверены")
    
    def _on_texture_select(self, event):
        """Handle texture selection."""
        selection = self.texture_listbox.curselection()
        if selection:
            index = selection[0]
            texture = self.textures[index]
            self._display_texture_details(texture)
    
    def _display_texture_details(self, texture: TextureData):
        """Display texture details."""
        detected = ', '.join(texture.detected_details) if texture.detected_details else 'Нет'
        missing = ', '.join(texture.missing_details) if texture.missing_details else 'Нет'
        
        self.detected_label.config(
            text=f"Найдено: {detected}",
            style='Valid.TLabel'
        )
        
        self.missing_label.config(
            text=f"Отсутствует: {missing}",
            style='Invalid.TLabel' if texture.missing_details else 'Valid.TLabel'
        )
        
        self.texture_validity_label.config(
            text=f"Статус: {'✓ Все детали на месте' if texture.is_valid else '✗ Отсутствуют детали'}",
            style='Valid.TLabel' if texture.is_valid else 'Invalid.TLabel'
        )
        
        # Display placeholder for image
        if PIL_AVAILABLE and texture.image_path and Path(texture.image_path).exists():
            self._display_image(texture.image_path)
        else:
            self.image_label.config(text="Нет изображения\n(Сгенерируйте или загрузите)")
    
    def _display_image(self, image_path: str):
        """Display an image in the image frame."""
        try:
            img = Image.open(image_path)
            img.thumbnail((400, 400))
            photo = ImageTk.PhotoImage(img)
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo  # Keep reference
        except Exception as e:
            logger.error(f"Error displaying image: {e}")
            self.image_label.config(text="Ошибка загрузки изображения")
    
    def _load_texture_image(self):
        """Load a texture image from file."""
        file_path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")]
        )
        
        if file_path:
            # Create mock texture data for loaded image
            texture_id = f"loaded_{Path(file_path).stem}"
            texture_data = TextureData(
                texture_id=texture_id,
                image_path=file_path,
                prompt="Загруженное изображение",
                required_details=[],
                detected_details=[],
                missing_details=[],
                is_valid=True
            )
            
            self.textures.append(texture_data)
            self._update_texture_list()
            self._select_texture(texture_id)
            self._display_image(file_path)
            
            self.status_bar.config(text=f"Изображение загружено: {file_path}")
    
    def _update_texture_list(self):
        """Update the texture listbox."""
        self.texture_listbox.delete(0, tk.END)
        for texture in self.textures:
            status = "✓" if texture.is_valid else "✗"
            self.texture_listbox.insert(tk.END, f"{status} {texture.texture_id}")
    
    def _select_texture(self, texture_id: str):
        """Select a texture by ID."""
        for i, texture in enumerate(self.textures):
            if texture.texture_id == texture_id:
                self.texture_listbox.selection_clear(0, tk.END)
                self.texture_listbox.selection_set(i)
                self._display_texture_details(texture)
                break
    
    # Overview methods
    def _refresh_overview(self):
        """Refresh the overview tab."""
        # Clear existing items
        for item in self.zones_tree.get_children():
            self.zones_tree.delete(item)
        for item in self.textures_tree.get_children():
            self.textures_tree.delete(item)
        
        # Populate zones
        valid_zones = 0
        for zone in self.zones:
            status = "✓ Валидна" if zone.is_valid else "✗ Невалидна"
            if zone.is_valid:
                valid_zones += 1
            
            self.zones_tree.insert('', 'end', values=(
                zone.zone_id,
                zone.timestamp or 'N/A',
                f"{zone.consistency_score:.2f}",
                f"{zone.temporal_score:.2f}",
                status
            ))
        
        # Populate textures
        valid_textures = 0
        for texture in self.textures:
            status = "✓ Валидна" if texture.is_valid else "✗ Невалидна"
            if texture.is_valid:
                valid_textures += 1
            
            detected = ', '.join(texture.detected_details[:3])
            if len(texture.detected_details) > 3:
                detected += "..."
            
            missing = ', '.join(texture.missing_details[:3])
            if len(texture.missing_details) > 3:
                missing += "..."
            
            self.textures_tree.insert('', 'end', values=(
                texture.texture_id,
                detected or 'Нет',
                missing or 'Нет',
                status
            ))
        
        # Update summary
        self.total_zones_label.config(text=f"Всего зон: {len(self.zones)}")
        self.valid_zones_label.config(text=f"Валидные зоны: {valid_zones}")
        self.total_textures_label.config(text=f"Всего текстур: {len(self.textures)}")
        self.valid_textures_label.config(text=f"Валидные текстуры: {valid_textures}")
        
        self.status_bar.config(text="Обзор обновлен")
    
    # Menu handlers
    def _save_project(self):
        """Save project to file."""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            # In real implementation, serialize data to JSON
            messagebox.showinfo("Информация", "Функция сохранения будет реализована в следующей версии")
            self.status_bar.config(text="Сохранение не реализовано")
    
    def _load_project(self):
        """Load project from file."""
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            # In real implementation, deserialize data from JSON
            messagebox.showinfo("Информация", "Функция загрузки будет реализована в следующей версии")
            self.status_bar.config(text="Загрузка не реализована")
    
    def _show_about(self):
        """Show about dialog."""
        about_text = """
Генерация контента с контролем целостности
        
Версия: 1.0.0
        
Эта система обеспечивает:
• Генерацию текстовых описаний зон с LLM
• Генерацию 2D-текстур через Stable Diffusion
• Контроль целостности на этапе создания
• Проверку непротиворечивости и временной логики
• Детекцию обязательных деталей в текстурах
        
Технологии:
• LLM: LLaMA/Mistral
• Текстуры: Stable Diffusion + ControlNet
        """
        
        messagebox.showinfo("О программе", about_text)


def run_visualizer():
    """Run the graphical visualizer application."""
    root = tk.Tk()
    app = ContentGenerationVisualizer(root)
    root.mainloop()


if __name__ == "__main__":
    run_visualizer()
