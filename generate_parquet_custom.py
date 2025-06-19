import argparse
import math
import os
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import time
import random
import string
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
from functools import lru_cache

try:
    from mimesis import Person, Address
    from mimesis.locales import Locale
    MIMESIS_AVAILABLE = True
except ImportError:
    MIMESIS_AVAILABLE = False

try:
    from faker import Faker
    FAKER_AVAILABLE = True
except ImportError:
    Faker = None
    FAKER_AVAILABLE = False

def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.2f} s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes} min {remaining_seconds:.2f} s"
    else:
        hours = int(seconds // 3600)
        remaining = seconds % 3600
        minutes = int(remaining // 60)
        remaining_seconds = remaining % 60
        return f"{hours} h {minutes} min {remaining_seconds:.2f} s"

def generate_column_mimesis(column_index, batch_size, cache_size, seed):
    random.seed(seed)

    person = Person(locale=Locale.ES)
    address = Address(locale=Locale.ES)

    is_name = column_index % 2 == 0

    cache = []
    for _ in range(cache_size):
        if is_name:
            cache.append(person.full_name())
        else:
            cache.append(address.address())

    column_data = []
    for _ in range(batch_size):
        if random.random() < 0.7 and cache:
            column_data.append(random.choice(cache))
        else:
            if is_name:
                column_data.append(person.full_name())
            else:
                column_data.append(address.address())

    return f'col_{column_index+1}', column_data

def generate_column_faker(column_index, batch_size, cache_size, seed):
    random.seed(seed)

    fake = Faker('es_ES')
    fake.seed_instance(seed)

    is_name = column_index % 2 == 0

    cache = []
    for _ in range(cache_size):
        if is_name:
            cache.append(fake.name())
        else:
            cache.append(fake.address())

    column_data = []
    for _ in range(batch_size):
        if random.random() < 0.7 and cache:
            column_data.append(random.choice(cache))
        else:
            if is_name:
                column_data.append(fake.name())
            else:
                column_data.append(fake.address())

    return f'col_{column_index+1}', column_data

def generate_column_random(column_index, batch_size, seed):
    random.seed(seed)
    np.random.seed(seed)

    k = 10
    chars = string.ascii_letters

    random_indices = np.random.randint(0, len(chars), size=(batch_size, k))
    column_data = [''.join(chars[idx] for idx in row) for row in random_indices]

    return f'col_{column_index+1}', column_data

def generate_mimesis_data_parallel(batch_size, columns, seed, max_workers=None, cache_size=1000):
    if max_workers is None:
        max_workers = min(columns, multiprocessing.cpu_count())

    actual_cache_size = min(batch_size // 10, cache_size)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(columns):
            column_seed = seed + i
            futures.append(executor.submit(
                generate_column_mimesis, i, batch_size, actual_cache_size, column_seed
            ))

        data = {}
        for future in futures:
            col_name, col_values = future.result()
            data[col_name] = col_values

    return data

def generate_faker_data_parallel(batch_size, columns, seed, max_workers=None, cache_size=1000):
    if max_workers is None:
        max_workers = min(columns, multiprocessing.cpu_count())

    actual_cache_size = min(batch_size // 10, cache_size)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(columns):
            column_seed = seed + i
            futures.append(executor.submit(
                generate_column_faker, i, batch_size, actual_cache_size, column_seed
            ))

        data = {}
        for future in futures:
            col_name, col_values = future.result()
            data[col_name] = col_values

    return data

def generate_random_data_parallel(batch_size, columns, seed, max_workers=None):
    if max_workers is None:
        max_workers = min(columns, multiprocessing.cpu_count())

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(columns):
            column_seed = seed + i
            futures.append(executor.submit(
                generate_column_random, i, batch_size, column_seed
            ))

        data = {}
        for future in futures:
            col_name, col_values = future.result()
            data[col_name] = col_values

    return data

def generate_faker_data_with_cache(batch_size, columns, fake, cache_size=1000):
    actual_cache_size = min(batch_size // 10, cache_size)

    name_cache = [fake.name() for _ in range(actual_cache_size)]
    address_cache = [fake.address() for _ in range(actual_cache_size)]

    data = {}
    for i in range(columns):
        column_data = []
        if i % 2 == 0:
            for _ in range(batch_size):
                if random.random() < 0.7:
                    column_data.append(random.choice(name_cache))
                else:
                    column_data.append(fake.name())
        else:
            for _ in range(batch_size):
                if random.random() < 0.7:
                    column_data.append(random.choice(address_cache))
                else:
                    column_data.append(fake.address())

        data[f'col_{i+1}'] = column_data

    return data

def generate_random_data_with_cache(batch_size, columns, seed, cache_size=1000):
    random.seed(seed)
    np.random.seed(seed)

    data = {}
    k = 10
    chars = string.ascii_letters

    actual_cache_size = min(batch_size // 10, cache_size)

    for i in range(columns):
        string_cache = []
        for _ in range(actual_cache_size):
            random_indices = np.random.randint(0, len(chars), size=k)
            string_cache.append(''.join(chars[idx] for idx in random_indices))

        column_data = []
        for _ in range(batch_size):
            if random.random() < 0.7 and string_cache:
                column_data.append(random.choice(string_cache))
            else:
                random_indices = np.random.randint(0, len(chars), size=k)
                column_data.append(''.join(chars[idx] for idx in random_indices))

        data[f'col_{i+1}'] = column_data

    return data

def generate_random_data(batch_size, columns):
    data = {}
    k = 10
    chars = string.ascii_letters

    for i in range(columns):
        random_indices = np.random.randint(0, len(chars), size=(batch_size, k))
        data[f'col_{i+1}'] = [''.join(chars[idx] for idx in row) for row in random_indices]

    return data

def process_batch(args):
    index, batch_size, columns, mode, seed, column_parallelism, cache_size = args

    start_time = time.time()

    random.seed(seed)
    np.random.seed(seed)

    if mode == 'mimesis' and MIMESIS_AVAILABLE:
        if column_parallelism:
            data = generate_mimesis_data_parallel(batch_size, columns, seed, cache_size=cache_size)
        else:
            person = Person(locale=Locale.ES)
            address = Address(locale=Locale.ES)

            data = {}
            for i in range(columns):
                if i % 2 == 0:
                    data[f'col_{i+1}'] = [person.full_name() for _ in range(batch_size)]
                else:
                    data[f'col_{i+1}'] = [address.address() for _ in range(batch_size)]

    elif mode == 'faker' and FAKER_AVAILABLE:
        fake = Faker('es_ES')
        fake.seed_instance(seed)

        if column_parallelism:
            data = generate_faker_data_parallel(batch_size, columns, seed, cache_size=cache_size)
        else:
            data = generate_faker_data_with_cache(batch_size, columns, fake, cache_size=cache_size)

    else:
        if column_parallelism:
            data = generate_random_data_parallel(batch_size, columns, seed)
        elif cache_size > 0:
            data = generate_random_data_with_cache(batch_size, columns, seed, cache_size)
        else:
            data = generate_random_data(batch_size, columns)

    df = pd.DataFrame(data)
    table = pa.Table.from_pandas(df)

    end_time = time.time()
    return index, table, (end_time - start_time)

def main():
    parser = argparse.ArgumentParser(
        description='Genera archivos Parquet con datos sintéticos por lotes, mostrando resultados en orden de índice.'
    )
    parser.add_argument('--rows', type=int, default=10000, help='Cantidad total de filas (default: %(default)s).')
    parser.add_argument('--columns', type=int, default=5, help='Cantidad de columnas (default: %(default)s).')
    parser.add_argument('--batch-size', type=int, default=1000, help='Número de filas por lote (default: %(default)s).')
    parser.add_argument('--filename', type=str, default='output', help='Nombre base del archivo (se agrega ".parquet" si no lleva).')
    parser.add_argument('--threads', type=int, default=0, help='Cantidad de procesos (0 para autodetectar, default: %(default)s).')
    parser.add_argument('--output-dir', type=str, default='', help='Ruta de salida (default: carpeta actual).')
    parser.add_argument('--mode', choices=['mimesis', 'faker', 'random'], default='mimesis',
                        help='Modo de generación (mimesis, faker o random). (default: %(default)s).')
    parser.add_argument('--compression', choices=['none', 'snappy', 'gzip', 'brotli', 'zstd'],
                        default='snappy', help='Algoritmo de compresión (default: %(default)s).')
    parser.add_argument('--column-parallel', action='store_true',
                        help='Usar paralelización por columnas (funciona con todos los modos, aumenta rendimiento pero usa más memoria).')
    parser.add_argument('--cache-size', type=int, default=1000,
                        help='Tamaño máximo de caché para valores frecuentes (funciona con todos los modos, default: %(default)s).')

    args = parser.parse_args()

    if not args.filename.lower().endswith('.parquet'):
        args.filename += '.parquet'

    if args.threads <= 0:
        args.threads = multiprocessing.cpu_count()

    output_dir = args.output_dir if args.output_dir else os.getcwd()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    out_path = os.path.join(output_dir, args.filename)

    print("Opciones elegidas:")
    print("-------------------------------------------------------")
    print(f" - rows            = {args.rows}")
    print(f" - columns         = {args.columns}")
    print(f" - batch-size      = {args.batch_size}")
    print(f" - filename        = {args.filename}")
    print(f" - threads         = {args.threads}")
    print(f" - mode            = {args.mode}")
    print(f" - output-dir      = {output_dir}")
    print(f" - compression     = {args.compression}")
    print(f" - column-parallel = {args.column_parallel}")
    print(f" - cache-size      = {args.cache_size}")
    print("-------------------------------------------------------")

    if args.mode == 'mimesis' and not MIMESIS_AVAILABLE:
        print("Advertencia: Mimesis no está instalado. Intentando con Faker...")
        args.mode = 'faker'

    if args.mode == 'faker' and not FAKER_AVAILABLE:
        print("Advertencia: Faker no está instalado. Usando generación aleatoria básica.")
        args.mode = 'random'

    print(f"Usando generador de datos: {args.mode}")

    optimizations = []
    if args.column_parallel:
        optimizations.append("paralelización por columnas")
    if args.cache_size > 0:
        optimizations.append(f"caché de valores (tamaño: {args.cache_size})")

    print(f"Optimizaciones activas: {', '.join(optimizations) if optimizations else 'ninguna'}")

    total_batches = math.ceil(args.rows / args.batch_size)
    print(f'Total de lotes a generar: {total_batches}')

    start_global = time.time()
    batch_times = []

    process_args = []
    base_seed = random.randint(1, 1000000)

    for i in range(total_batches):
        rows_in_this_batch = min(args.batch_size, args.rows - i * args.batch_size)
        process_args.append((
            i, rows_in_this_batch, args.columns, args.mode, 
            base_seed + i, args.column_parallel, args.cache_size
        ))

    with ProcessPoolExecutor(max_workers=args.threads) as executor:
        futures = [executor.submit(process_batch, arg) for arg in process_args]

        results = [None] * total_batches
        completed = 0

        for future in futures:
            batch_index, table, lote_time = future.result()
            results[batch_index] = (table, lote_time)
            batch_times.append(lote_time)

            completed += 1
            percent = (completed / total_batches) * 100
            print(
                f'Lote {batch_index+1}/{total_batches} completado ({percent:.2f}%). '
                f'Tiempo del lote: {format_time(lote_time)}'
            )

    tables = [result[0] for result in results]

    compression = args.compression if args.compression != 'none' else None

    print("Combinando tablas y escribiendo archivo Parquet...")
    combined_table = pa.concat_tables(tables)

    pq.write_table(
        combined_table, 
        out_path,
        compression=compression,
        use_dictionary=True,
        write_statistics=True
    )

    total_time = time.time() - start_global
    print("-------------------------------------------------------")
    print("Resumen estadístico:")
    print(f" - Tiempo total: {format_time(total_time)}")
    if batch_times:
        avg_time = sum(batch_times)/len(batch_times)
        min_time = min(batch_times)
        max_time = max(batch_times)
        print(f" - Promedio por lote: {format_time(avg_time)}")
        print(f" - Lote más rápido: {format_time(min_time)}")
        print(f" - Lote más lento: {format_time(max_time)}")
    print("-------------------------------------------------------")
    print(f'Archivo Parquet generado en: {out_path}')

if __name__ == '__main__':
    main()
