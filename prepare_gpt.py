# file: prepare_gpt.py
# directory: .
import os


def parse_directory_structure(structure_lines):
    file_paths = []
    dir_stack = []

    for line in structure_lines:
        stripped_line = line.rstrip('\n')
        # Определяем уровень вложенности по количеству отступов
        indent_level = len(line) - len(line.lstrip(' '))
        current_dir = line.strip()

        if not current_dir:
            continue  # Пропускаем пустые строки

        # Проверяем, является ли это директорией или файлом
        if current_dir.endswith('/'):
            # Это директория
            dir_name = current_dir.rstrip('/')
            # Обновляем стек директорий до текущего уровня вложенности
            dir_stack = dir_stack[:indent_level // 4]
            dir_stack.append(dir_name)
        else:
            # Это файл
            # Полный путь к файлу
            dir_stack_copy = dir_stack[:]
            dir_stack_copy = dir_stack_copy[:indent_level // 4]
            file_name = current_dir
            file_path = os.path.join(*dir_stack_copy, file_name)
            file_paths.append(file_path)
    return file_paths


def main():
    # Читаем содержимое файла directory_structure.txt
    try:
        with open('directory_struct.txt', 'r', encoding='utf-8') as f:
            directory_structure = f.read()
    except Exception as e:
        print(f"Ошибка при чтении файла directory_struct.txt: {e}")
        return

    # Читаем структуру из report_list.txt
    try:
        with open('report_list.txt', 'r', encoding='utf-8') as f:
            structure_lines = f.readlines()
    except Exception as e:
        print(f"Ошибка при чтении файла report_list.txt: {e}")
        return

    # Парсим структуру и получаем список файлов
    file_list = parse_directory_structure(structure_lines)

    # Создаём и открываем файл report.txt для записи
    try:
        with open('report.txt', 'w', encoding='utf-8') as report_file:
            # Записываем содержимое directory_structure.txt
            report_file.write(directory_structure)

            # Для каждого файла из списка
            for file_path in file_list:
                # Проверяем, существует ли файл
                if not os.path.isfile(file_path):
                    print(f"Файл {file_path} не найден, пропускаем.")
                    continue

                # Добавляем разделитель
                report_file.write('\n----\n\n')

                # Читаем содержимое файла и записываем в report.txt
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    report_file.write(content)
                except Exception as e:
                    print(f"Ошибка при чтении файла {file_path}: {e}")
                    continue

        print("Файл report.txt успешно создан.")
    except Exception as e:
        print(f"Ошибка при создании файла report.txt: {e}")


if __name__ == '__main__':
    main()
