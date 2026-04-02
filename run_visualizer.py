#!/usr/bin/env python3
"""
Launch script for the Content Generation Visualizer

Run this script to start the graphical interface.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from visualizer import run_visualizer

if __name__ == "__main__":
    print("Запуск графического интерфейса...")
    print("=" * 50)
    print("Генерация контента с контролем целостности")
    print("=" * 50)
    print()
    print("Функции:")
    print("  📝 Генерация текстовых описаний зон")
    print("  🎨 Генерация 2D-текстур")
    print("  📊 Обзор целостности и валидации")
    print()
    print("Закройте окно для выхода.")
    print("=" * 50)
    
    try:
        run_visualizer()
    except KeyboardInterrupt:
        print("\nПриложение закрыто пользователем")
        sys.exit(0)
    except Exception as e:
        print(f"\nОшибка: {e}")
        sys.exit(1)
