import struct
import numpy as np
import tensorflow as tf
from scipy.linalg import hadamard
from sklearn.cluster import KMeans
import heapq
from collections import Counter

# UTILITAIRES BINAIRES  (inspirés de ZN)

def bits_to_bytes(bitstring):
    
    pad = (8 - len(bitstring) % 8) % 8
    padded = bitstring + '0' * pad
    b = bytearray()
    for i in range(0, len(padded), 8):
        b.append(int(padded[i:i + 8], 2))
    return bytes(b), pad

def bytes_to_bits(data, pad):
    #Dépacke des octets en chaîne de bits, en retirant le padding
    bits = ''.join(f'{byte:08b}' for byte in data)
    return bits if pad == 0 else bits[:-pad]


#HADAMARD 

def apply_hadamard_block(W, block_size=256):
    H = hadamard(block_size) / np.sqrt(block_size)
    shape_orig = W.shape
    W_flat = W.flatten()
    n = len(W_flat)
    pad_size = (block_size - (n % block_size)) % block_size
    W_padded = np.pad(W_flat, (0, pad_size), 'constant')
    W_blocks = W_padded.reshape(-1, block_size)
    W_transformed = W_blocks @ H.T
    return W_transformed, shape_orig, pad_size

def inverse_hadamard_block(W_transformed, shape_orig, pad_size, block_size=256):
    H = hadamard(block_size) / np.sqrt(block_size)
    W_flat_rec = (W_transformed @ H.T).flatten()
    if pad_size > 0:
        W_flat_rec = W_flat_rec[:-pad_size]
    return W_flat_rec.reshape(shape_orig)


#SEUILLAGE DEADZONE

def deadzone_threshold(W_hadamard, threshold):
    """Force à 0.0 tous les coefficients dont |val| < threshold."""
    W_thresh = W_hadamard.copy()
    W_thresh[np.abs(W_thresh) < threshold] = 0.0
    return W_thresh

def compute_deadzone_threshold(W_hadamard, sparsity=0.50):
    return float(np.percentile(np.abs(W_hadamard), sparsity * 100))


#QUANTIFICATION K-MEANS (cluster 0 reserve aux zeros)

def kmeans_quantize_with_zero(W_flat, k_clusters=16):
    """
    K-Means avec cluster 0 = centroïde 0.0 (zéros deadzone).
    Les k-1 autres centroïdes sont appris sur les valeurs non-nulles.
    """
    zero_mask = (W_flat == 0.0)
    nonzero_vals = W_flat[~zero_mask]
    k_nonzero = min(k_clusters - 1, max(1, len(nonzero_vals)))

    kmeans = KMeans(n_clusters=k_nonzero, n_init=3, random_state=42)
    nz_indices = kmeans.fit_predict(nonzero_vals.reshape(-1, 1))
    nz_centroids = kmeans.cluster_centers_.flatten()

    centroids = np.concatenate([[0.0], nz_centroids])   # centroïde 0 = zéro exact
    indices = np.zeros(len(W_flat), dtype=np.int32)
    indices[~zero_mask] = nz_indices + 1
    return indices, centroids

def inverse_kmeans(indices, centroids, shape):
    return centroids[indices].reshape(shape)


#RLE : Run-Length Encoding sur les indices K-Means

BITS_VAL = 4    # log2(16 clusters)
BITS_RUN = 12   # runs jusqu'à 4095 coefficients consécutifs

def rle_encode(indices):
    """
    Encode une séquence d'indices en paires (val_idx, run_length).
    Les runs > 2^BITS_RUN - 1 sont découpés en plusieurs paires.
    Retourne une liste de tuples (int, int).
    """
    if len(indices) == 0:
        return []
    max_run = (1 << BITS_RUN) - 1
    rle = []
    current = int(indices[0])
    count = 1
    for val in indices[1:]:
        v = int(val)
        if v == current:
            count += 1
            if count > max_run:          # découpe les très longs runs
                rle.append((current, max_run))
                count = 1
        else:
            rle.append((current, count))
            current = v
            count = 1
    rle.append((current, count))
    return rle

def rle_decode(rle_pairs, total_length):
    """Reconstruit la séquence d'indices à partir des paires RLE."""
    indices = np.empty(total_length, dtype=np.int32)
    pos = 0
    for val, count in rle_pairs:
        indices[pos:pos + count] = val
        pos += count
    return indices

def rle_pairs_to_bitstring(rle_pairs):
    """
    Sérialise les paires RLE en une chaîne binaire brute
    (avant Huffman) : chaque paire : BITS_VAL + BITS_RUN bits.
    """
    bits = ''
    for val, run in rle_pairs:
        bits += format(val, f'0{BITS_VAL}b')
        bits += format(run, f'0{BITS_RUN}b')
    return bits

def rle_pairs_from_bitstring(bits, num_pairs):
    """Désérialise une chaîne binaire brute en paires RLE."""
    pair_bits = BITS_VAL + BITS_RUN
    pairs = []
    for i in range(num_pairs):
        chunk = bits[i * pair_bits:(i + 1) * pair_bits]
        val = int(chunk[:BITS_VAL], 2)
        run = int(chunk[BITS_VAL:], 2)
        pairs.append((val, run))
    return pairs


#HUFFMAN sur les paires RLE

class HuffmanNode:
    def __init__(self, symbol=None, freq=0, left=None, right=None):
        self.symbol = symbol
        self.freq   = freq
        self.left   = left
        self.right  = right
    def __lt__(self, other):
        return self.freq < other.freq

def _build_huffman_tree(symbols, freqs):
    """Construit l'arbre Huffman (même logique que test.py)."""
    heap = [HuffmanNode(sym, f) for sym, f in zip(symbols, freqs)]
    heapq.heapify(heap)
    if len(heap) == 1:                        # cas dégénéré : un seul symbole
        node = heap[0]
        return HuffmanNode(left=node, right=HuffmanNode(symbol=None, freq=0))
    while len(heap) > 1:
        l, r = heapq.heappop(heap), heapq.heappop(heap)
        heapq.heappush(heap, HuffmanNode(freq=l.freq + r.freq, left=l, right=r))
    return heap[0]

def _generate_codes(node, prefix='', codebook=None):
    """Génère le codebook {symbole: bitstring} (même logique que test.py)."""
    if codebook is None:
        codebook = {}
    if node is None:
        return codebook
    if node.symbol is not None:
        codebook[node.symbol] = prefix or '0'
    _generate_codes(node.left,  prefix + '0', codebook)
    _generate_codes(node.right, prefix + '1', codebook)
    return codebook

# ---- Sérialisation du codebook ----
# Format binaire compact pour stocker la table de décodage :
#   [2 octets] nb_symboles
#   Pour chaque symbole :
#     [1 octet ] val_idx  (0–15)
#     [2 octets] run      (0–4095, stocké sur 2 octets pour simplicité)
#     [1 octet ] longueur du code Huffman (nb de bits)
#   → 4 octets par symbole unique
# Total codebook : 2 + 4 × nb_symboles_uniques octets

def _make_canonical_codebook(codebook):
    """
    Transforme un codebook quelconque en codebook canonique :
    trie par (longueur_code, symbole) et réassigne des codes entiers croissants.
    Encode ET décode utilisent ce même codebook 
    """
    entries = sorted(codebook.keys(), key=lambda s: (len(codebook[s]), s))
    canonical = {}
    code = 0
    prev_len = 0
    for sym in entries:
        length = len(codebook[sym])
        code <<= (length - prev_len)
        canonical[sym] = format(code, f'0{length}b')
        code += 1
        prev_len = length
    return canonical

def serialize_codebook(codebook):
    """
    Sérialise le codebook canonique en bytes.
    Format : [2 B] nb_symboles | pour chaque symbole : val(1B) run(2B) code_len(1B)
    → 4 octets par symbole unique.
    """
    n = len(codebook)
    data = struct.pack('>H', n)
    for (val, run), code in codebook.items():
        data += struct.pack('>BBH', val, len(code), run)
    return data

def deserialize_and_rebuild(codebook_bytes):
    """
    Lit le codebook sérialisé et reconstruit le codebook canonique identique.
    Même tri (longueur, symbole) → mêmes codes que côté encodeur.
    """
    n = struct.unpack('>H', codebook_bytes[:2])[0]
    entries = []
    for i in range(n):
        offset = 2 + i * 4
        val, code_len, run = struct.unpack('>BBH', codebook_bytes[offset:offset + 4])
        entries.append(((val, run), code_len))
    # Reconstruction canonique (même ordre que _make_canonical_codebook)
    entries_sorted = sorted(entries, key=lambda x: (x[1], x[0]))
    codebook = {}
    code = 0
    prev_len = 0
    for symbol, length in entries_sorted:
        code <<= (length - prev_len)
        codebook[symbol] = format(code, f'0{length}b')
        code += 1
        prev_len = length
    return codebook


def huffman_encode_real(rle_pairs):
    """
    Encode les paires RLE avec Huffman canonique.
    Retourne les octets packés (comme to_bytes dans test.py).
    """
    freqs     = Counter(rle_pairs)
    symbols   = list(freqs.keys())
    freq_list = [freqs[s] for s in symbols]

    root     = _build_huffman_tree(symbols, freq_list)
    raw_cb   = _generate_codes(root)
    codebook = _make_canonical_codebook(raw_cb)   # codes canoniques reproductibles

    # Bitstream : concaténation des codes (même logique que weights_2_bin dans test.py)
    bitstring = ''.join(codebook[pair] for pair in rle_pairs)

    # Packing réel en octets (même logique que to_bytes dans test.py)
    compressed_bytes, padding = bits_to_bytes(bitstring)

    codebook_bytes = serialize_codebook(codebook)

    stats = {
        'nb_bits_huffman'  : len(bitstring),
        'nb_bytes_huffman' : len(compressed_bytes),
        'nb_bytes_codebook': len(codebook_bytes),
        'nb_symboles_uniq' : len(symbols),
    }
    return compressed_bytes, padding, codebook_bytes, stats

def huffman_decode_real(compressed_bytes, padding, codebook_bytes, num_pairs):
    """Décode le bitstream Huffman et retourne les paires RLE."""
    codebook   = deserialize_and_rebuild(codebook_bytes)
    reverse_cb = {code: sym for sym, code in codebook.items()}

    bitstring = bytes_to_bits(compressed_bytes, padding)

    rle_pairs = []
    buf = ''
    for bit in bitstring:
        buf += bit
        if buf in reverse_cb:
            rle_pairs.append(reverse_cb[buf])
            buf = ''
            if len(rle_pairs) == num_pairs:
                break
    return rle_pairs


#PIPELINE COMPLET PAR COUCHE

def compress_layer(W, k_clusters=16, deadzone_sparsity=0.50):
    """
    Pipeline complet :
      Hadamard → Deadzone → K-Means → RLE → Huffman (réel)
    Retourne les données de décompression + les métriques étape par étape.
    """
    B0 = W.size * 4   # octets originaux (float32)

    #Hadamard : même taille float32
    W_hadamard, shape_orig, pad_size = apply_hadamard_block(W)
    B1 = W_hadamard.size * 4   # identique à B0 (+ padding de bloc éventuel)

    # Deadzone :les zéros ne réduisent pas encore la taille brute,
    threshold    = compute_deadzone_threshold(W_hadamard, sparsity=deadzone_sparsity)
    W_thresh     = deadzone_threshold(W_hadamard, threshold)
    W_flat       = W_thresh.flatten()
    sparsity_pct = np.mean(W_flat == 0.0) * 100
    B2 = W_flat.size * 4       # toujours float32, mais {sparsity_pct}% sont 0.0

    #K-Means : chaque poids devient un indice sur ceil(log2(k)) bits
    indices, centroids = kmeans_quantize_with_zero(W_flat, k_clusters=k_clusters)
    bits_per_idx = int(np.ceil(np.log2(k_clusters)))       # 4 bits pour 16 clusters
    B3 = (len(indices) * bits_per_idx + 7) // 8 + len(centroids) * 4

    # RLE :les paires remplacent les runs de zéros répétés
    rle_pairs = rle_encode(indices)
    bits_per_pair = BITS_VAL + BITS_RUN                    # 4 + 12 = 16 bits/paire
    B4 = (len(rle_pairs) * bits_per_pair + 7) // 8 + len(centroids) * 4

    #Huffman :packing en octets + codebook
    compressed_bytes, padding, codebook_bytes, huff_stats = huffman_encode_real(rle_pairs)
    bytes_centroids = len(centroids) * 4
    bytes_codebook  = len(codebook_bytes)
    bytes_num_pairs = 4
    bytes_bitstream = len(compressed_bytes)
    B5 = bytes_centroids + bytes_codebook + bytes_num_pairs + bytes_bitstream

    stage_bytes = {
        'Original (float32)'         : B0,
        'Hadamard (float32, lossless)': B1,
        'Deadzone (float32 + zéros)' : B2,
        'K-Means (indices + centroides)': B3,
        'RLE (paires binaires + centroides)': B4,
        'Huffman (octets réels + codebook)' : B5,
    }

    return {
        # données de décompression
        'compressed_bytes' : compressed_bytes,
        'padding'          : padding,
        'codebook_bytes'   : codebook_bytes,
        'num_rle_pairs'    : len(rle_pairs),
        'centroids'        : centroids,
        'num_indices'      : len(indices),
        'shape_orig'       : shape_orig,
        'pad_size'         : pad_size,
        'W_hadamard_shape' : W_hadamard.shape,
        # métriques
        'bits_originaux'   : B0 * 8,
        'bits_comprimes'   : B5 * 8,
        'sparsity_pct'     : sparsity_pct,
        'threshold'        : threshold,
        'huff_stats'       : huff_stats,
        'stage_bytes'      : stage_bytes,
        'bytes_detail'     : {
            'centroids' : bytes_centroids,
            'codebook'  : bytes_codebook,
            'num_pairs' : bytes_num_pairs,
            'bitstream' : bytes_bitstream,
        },
    }

def decompress_layer(data):
    """
    Pipeline inverse 
    """
    rle_pairs   = huffman_decode_real(
                    data['compressed_bytes'], data['padding'],
                    data['codebook_bytes'],   data['num_rle_pairs'])
    indices     = rle_decode(rle_pairs, data['num_indices'])
    W_hadamard  = inverse_kmeans(indices, data['centroids'], data['W_hadamard_shape'])
    W_rec       = inverse_hadamard_block(W_hadamard, data['shape_orig'], data['pad_size'])
    return W_rec


# =========================================================
# 7. TEST SUR MNIST (donnees) ; LeNet-300-100 (reseau)
if __name__ == "__main__":
    K_CLUSTERS        = 16
    DEADZONE_SPARSITY = 0.50

    print("=" * 65)
    print("  LeNet-300-100 sur MNIST — Pipeline de compression")
    print(f"  Hadamard → Deadzone({DEADZONE_SPARSITY*100:.0f}%) → K-Means({K_CLUSTERS}) → RLE → Huffman")
    print(f"  RLE : {BITS_VAL} bits/val + {BITS_RUN} bits/run  |  Huffman : vrais octets")
    print("=" * 65)

    print("\n--- 1. Entraînement LeNet-300-100 ---")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(300, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(10,  activation='softmax'),
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5, batch_size=128, verbose=1)

    _, acc_orig = model.evaluate(x_test, y_test, verbose=0)

    W1, b1 = model.layers[1].get_weights()
    W2, b2 = model.layers[2].get_weights()
    W3, b3 = model.layers[3].get_weights()

    #Taille réelle du réseau complet (poids + biais, toutes couches) 
    def layer_bytes(layer):
        return sum(w.nbytes for w in layer.get_weights())

    total_model_bytes = sum(layer_bytes(l) for l in model.layers)
    print(f"\n{'='*65}")
    print(f"  TAILLE RÉELLE DU RÉSEAU (avant compression)")
    print(f"{'='*65}")
    for i, layer in enumerate(model.layers):
        ws = layer.get_weights()
        if not ws:
            continue
        nb = sum(w.nbytes for w in ws)
        shapes = ' + '.join(str(w.shape) for w in ws)
        print(f"  Layer {i} ({layer.name:<20}) : {shapes:<28} = {nb:>8} octets")
    print(f"  {'─'*61}")
    print(f"  Total réseau complet                               = {total_model_bytes:>8} octets")
    print(f"  (couches 1+2 compressibles seulement)              = {(W1.size+W2.size)*4:>8} octets")
    print(f"{'='*65}")

    #Compression étape par étape 
    print("\n--- 2. Compression (étape par étape) ---")

    def print_stage_table(data, layer_name):
        B0   = data['stage_bytes']['Original (float32)']
        print(f"\n  {'─'*61}")
        print(f"  Couche {layer_name}")
        print(f"  {'─'*61}")
        print(f"  {'Étape':<40} {'Octets':>9}  {'Ratio':>7}  {'Gain étape':>10}")
        print(f"  {'─'*61}")
        prev = None
        for nom, octets in data['stage_bytes'].items():
            ratio_vs_orig  = B0 / octets if octets > 0 else float('inf')
            gain_vs_prev   = f"{prev/octets:.2f}×" if prev and prev != octets else "  —  "
            marker = " ◄" if nom.startswith('Huffman') else ""
            print(f"  {nom:<40} {octets:>9,}  {ratio_vs_orig:>6.2f}×  {gain_vs_prev:>10}{marker}")
            prev = octets
        print(f"  {'─'*61}")

    data1 = compress_layer(W1, k_clusters=K_CLUSTERS, deadzone_sparsity=DEADZONE_SPARSITY)
    print_stage_table(data1, "1  —  784×300  (235 200 poids)")
    print(f"  Sparsité deadzone : {data1['sparsity_pct']:.1f}%  |  "
          f"Paires RLE : {data1['num_rle_pairs']:,} / {data1['num_indices']:,}  |  "
          f"Symboles Huffman : {data1['huff_stats']['nb_symboles_uniq']}")

    data2 = compress_layer(W2, k_clusters=K_CLUSTERS, deadzone_sparsity=DEADZONE_SPARSITY)
    print_stage_table(data2, "2  —  300×100  ( 30 000 poids)")
    print(f"  Sparsité deadzone : {data2['sparsity_pct']:.1f}%  |  "
          f"Paires RLE : {data2['num_rle_pairs']:,} / {data2['num_indices']:,}  |  "
          f"Symboles Huffman : {data2['huff_stats']['nb_symboles_uniq']}")

    total_bytes_orig = (W1.size + W2.size) * 4
    total_bytes_comp = data1['stage_bytes']['Huffman (octets réels + codebook)'] \
                     + data2['stage_bytes']['Huffman (octets réels + codebook)']
    ratio_global = total_bytes_orig / total_bytes_comp

    print(f"\n  Total compressé (couches 1+2) : {total_bytes_comp:,} octets")
    print(f"  Ratio global                  : {ratio_global:.2f}×")

    print("\n--- 3. Décompression ---")
    W1_rec = decompress_layer(data1)
    W2_rec = decompress_layer(data2)

    print("\n--- 4. Évaluation du modèle décompressé ---")
    model.layers[1].set_weights([W1_rec, b1])
    model.layers[2].set_weights([W2_rec, b2])
    _, acc_dec = model.evaluate(x_test, y_test, verbose=0)
    delta = (acc_orig - acc_dec) * 100

    print(f"\n{'='*65}")
    print(f"  RÉSULTATS FINAUX")
    print(f"{'='*65}")
    print(f"  Précision originale    : {acc_orig*100:.2f}%")
    print(f"  Précision décompressée : {acc_dec*100:.2f}%")
    print(f"  Perte de précision     : {delta:+.2f}%")
    print(f"  Ratio de compression   : {ratio_global:.2f}×")
    print(f"{'='*65}")
