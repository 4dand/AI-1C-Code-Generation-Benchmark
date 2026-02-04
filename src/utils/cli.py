"""
cli utilities - форматирование cli
"""

def print_section(title: str):
    """Вывести заголовок секции"""
    print()
    print("=" * 60)
    print(f" {title}")
    print("=" * 60)


def print_kv(label: str, value: str, indent: int = 2):
    """Вывести пару ключ-значение с выравниванием"""
    print(" " * indent + f"{label}: {value}")


def print_table_header(*columns: tuple[str, int], indent: int = 2):
    """Вывести заголовок таблицы
    
    Args:
        columns: кортежи (название, ширина)
        indent: отступ слева
    """
    header = " | ".join(f"{col:<{width}}" for col, width in columns)
    print(" " * indent + header)
    total_width = sum(width for _, width in columns) + 3 * (len(columns) - 1)
    print(" " * indent + "-" * total_width)


def print_table_row(*values: tuple[str, int], indent: int = 2):
    """Вывести строку таблицы
    
    Args:
        values: кортежи (значение, ширина)
        indent: отступ слева
    """
    row = " | ".join(f"{val:<{width}}" for val, width in values)
    print(" " * indent + row)

