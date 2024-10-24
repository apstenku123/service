# file: update_headers.py
# directory: old_inter_backup
import os
import sys
import fnmatch
import re

# Определяем стили комментариев для различных расширений файлов
COMMENT_STYLES = {
    '.py': '#',
    '.html': '<!-- -->',
    '.htm': '<!-- -->',
    '.js': '//',
    '.css': '/* */',
    '.c': '//',
    '.cpp': '//',
    '.java': '//',
    '.php': '//',
    '.sh': '#',
    '.txt': '#',
    # Добавьте дополнительные расширения и стили комментариев по необходимости
}

# Загружаем фильтры из файла filter.txt
def load_filters(filter_file='filter.txt'):
    filters = []
    if os.path.exists(filter_file):
        with open(filter_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    filters.append(line)
    return filters

# Загружаем список расширений файлов для обработки
def load_extensions(extensions_file='extensions.txt'):
    extensions = []
    if os.path.exists(extensions_file):
        with open(extensions_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if not line.startswith('.'):
                        line = '.' + line
                    extensions.append(line.lower())
    else:
        # Если файл не найден, используем все поддерживаемые расширения
        extensions = list(COMMENT_STYLES.keys())
    return extensions

# Загружаем список расширений файлов для структуры директории
def load_structure_extensions(structure_extensions_file='structure_extensions.txt'):
    extensions = []
    if os.path.exists(structure_extensions_file):
        with open(structure_extensions_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if not line.startswith('.'):
                        line = '.' + line
                    extensions.append(line.lower())
    else:
        # Если файл не найден, включаем все файлы
        extensions = ['*']
    return extensions

# Проверяем, нужно ли пропустить файл или директорию
def should_skip(name, filters):
    for pattern in filters:
        if fnmatch.fnmatch(name, pattern):
            return True
    return False


# Проверяем и обновляем заголовки файлов
def check_and_update_file(file_path, root_dir, processing_extensions):
    file_name = os.path.basename(file_path)
    dir_name = os.path.relpath(os.path.dirname(file_path), root_dir)  # Измененная строка

    _, ext = os.path.splitext(file_name)
    ext = ext.lower()

    if ext not in processing_extensions:
        return

    comment_style = COMMENT_STYLES.get(ext)

    # Пропускаем файлы, для которых не определен стиль комментариев
    if not comment_style:
        return

    # Читаем содержимое файла
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Ошибка при чтении файла {file_path}: {e}")
        return

    changed = False

    # Подготавливаем правильные комментарии с метками
    if comment_style == '<!-- -->':
        file_comment = f'<!-- file: {file_name} -->\n'
        dir_comment = f'<!-- directory: {dir_name} -->\n'
        # Обновленное регулярное выражение для поиска старых и новых комментариев
        comment_regex = re.compile(r'<!--\s*(file:|directory:)[^>]*-->')
    elif comment_style == '/* */':
        file_comment = f'/* file: {file_name} */\n'
        dir_comment = f'/* directory: {dir_name} */\n'
        comment_regex = re.compile(r'/\*\s*(file:|directory:)[^\*]*\*/')
    else:
        # Предполагаем однострочный стиль комментариев (например, # или //)
        file_comment = f'{comment_style} file: {file_name}\n'
        dir_comment = f'{comment_style} directory: {dir_name}\n'
        escaped_comment_style = re.escape(comment_style)
        comment_regex = re.compile(rf'{escaped_comment_style}\s*(file:|directory:).*')

    # Удаляем все комментарии в начале файла, соответствующие нашему шаблону
    indices_to_remove = []
    for i, line in enumerate(lines):
        if i > 10:
            break  # Проверяем только первые 10 строк
        if comment_regex.match(line.strip()):
            indices_to_remove.append(i)
        else:
            # Останавливаемся, когда встречаем первую не соответствующую строку
            break

    # Удаляем найденные комментарии
    if indices_to_remove:
        for index in reversed(indices_to_remove):
            del lines[index]
        changed = True

    # Добавляем обновленные комментарии в начало файла
    lines.insert(0, dir_comment)
    lines.insert(0, file_comment)
    changed = True

    # Записываем изменения обратно в файл
    if changed:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            print(f"Обновлен файл: {file_path}")
        except Exception as e:
            print(f"Ошибка при записи файла {file_path}: {e}")


# Генерируем структуру директории
def generate_directory_structure(root_dir, filters, structure_extensions):
    structure_lines = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Относительный путь
        rel_path = os.path.relpath(dirpath, root_dir)
        if should_skip(rel_path, filters):
            dirnames.clear()  # Не заходим в эту директорию
            continue

        # Фильтруем директории и файлы
        dirnames[:] = [d for d in dirnames if not should_skip(os.path.join(rel_path, d), filters) and not d.startswith('.')]
        filenames = [f for f in filenames if not should_skip(f, filters) and not f.startswith('.')]

        # Пропускаем временные файлы и директории Python
        dirnames[:] = [d for d in dirnames if d != '__pycache__']
        filenames = [f for f in filenames if not f.endswith('.pyc')]

        level = rel_path.count(os.sep)
        indent = ' ' * 4 * level
        dir_name = os.path.basename(dirpath)
        structure_lines.append(f'{indent}{dir_name}/\n')
        subindent = ' ' * 4 * (level + 1)
        for f in filenames:
            # Фильтруем файлы по расширению для структуры
            _, ext = os.path.splitext(f)
            ext = ext.lower()
            if '*' in structure_extensions or ext in structure_extensions:
                structure_lines.append(f'{subindent}{f}\n')

    try:
        with open('directory_struct.txt', 'w', encoding='utf-8') as f:
            f.writelines(structure_lines)
        print("Файл directory_struct.txt успешно создан.")
    except Exception as e:
        print(f"Ошибка при записи файла directory_struct.txt: {e}")

def main():
    root_dir = os.getcwd()
    filters = load_filters()
    processing_extensions = load_extensions()
    structure_extensions = load_structure_extensions()

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Относительный путь
        rel_path = os.path.relpath(dirpath, root_dir)
        if should_skip(rel_path, filters):
            dirnames.clear()  # Не заходим в эту директорию
            continue

        # Фильтруем директории
        dirnames[:] = [d for d in dirnames if not should_skip(os.path.join(rel_path, d), filters) and not d.startswith('.') and d != '__pycache__']

        for filename in filenames:
            if should_skip(filename, filters) or filename.startswith('.'):
                continue
            if filename.endswith('.pyc'):
                continue
            file_path = os.path.join(dirpath, filename)
            check_and_update_file(file_path, root_dir, processing_extensions)

    generate_directory_structure(root_dir, filters, structure_extensions)
    print("Обработка завершена.")

if __name__ == '__main__':
    main()
