import base64
import gzip
import numpy as np
from io import BytesIO

def decode_matrix(base64_string, rows, cols):
    # Decodificar base64
    compressed_bytes = base64.b64decode(base64_string)
    # Descomprimir gzip
    with gzip.GzipFile(fileobj=BytesIO(compressed_bytes)) as f:
        decompressed_bytes = f.read()
    # Interpretar como array de uint16 (little endian)
    flat_array = np.frombuffer(decompressed_bytes, dtype='<u2')  # '<u2' = little-endian uint16
    # Reshape a matriz
    matrix = flat_array.reshape((rows, cols))

    return matrix