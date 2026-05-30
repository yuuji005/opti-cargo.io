import random
import math
import matplotlib.pyplot as plt

# ==========================================
# 1. PERSIAPAN DATA (PROBLEM INSTANCE)
# ==========================================
BARANG = [
    {"nama": "Genset Mini", "berat": 15, "nilai": 500},
    {"nama": "Beras 1 Karung", "berat": 25, "nilai": 350},
    {"nama": "Suku Cadang", "berat": 10, "nilai": 250},
    {"nama": "Kotak P3K", "berat": 2, "nilai": 150},
    {"nama": "Tenda Pleton", "berat": 20, "nilai": 400},
    {"nama": "Air Mineral", "berat": 10, "nilai": 100},
    {"nama": "Radio Komunikasi", "berat": 3, "nilai": 300},
    {"nama": "Makanan Kaleng", "berat": 8, "nilai": 200},
]

KAPASITAS_MAKSIMAL = 50  
JUMLAH_BARANG = len(BARANG)

def hitung_fitness(solusi):
    total_berat = 0
    total_nilai = 0
    for i in range(JUMLAH_BARANG):
        if solusi[i] == 1:
            total_berat += BARANG[i]["berat"]
            total_nilai += BARANG[i]["nilai"]
            
    if total_berat > KAPASITAS_MAKSIMAL:
        return 0 
    return total_nilai

def generate_solusi_awal():
    return [random.choice([0, 1]) for _ in range(JUMLAH_BARANG)]

def get_neighbors(solusi):
    neighbors = []
    for i in range(JUMLAH_BARANG):
        tetangga_baru = solusi.copy()
        tetangga_baru[i] = 1 - tetangga_baru[i] 
        neighbors.append(tetangga_baru)
    return neighbors

# ==========================================
# 2. HILL CLIMBING & SIMULATED ANNEALING
# ==========================================
def steepest_ascent_hill_climbing(max_iter=50):
    current = generate_solusi_awal()
    current_fitness = hitung_fitness(current)
    history = [current_fitness]
    
    for _ in range(max_iter):
        neighbors = get_neighbors(current)
        best_neighbor = current
        best_neighbor_fitness = current_fitness
        
        for neighbor in neighbors:
            neighbor_fitness = hitung_fitness(neighbor)
            if neighbor_fitness > best_neighbor_fitness:
                best_neighbor = neighbor
                best_neighbor_fitness = neighbor_fitness
                
        if best_neighbor_fitness <= current_fitness:
            history.append(current_fitness) 
            continue
            
        current = best_neighbor
        current_fitness = best_neighbor_fitness
        history.append(current_fitness)
        
    return current, current_fitness, history

def simulated_annealing(T0=100.0, cooling_rate=0.9, T_min=1.0, iter_per_temp=1):
    current = generate_solusi_awal()
    current_fitness = hitung_fitness(current)
    history_fitness = [current_fitness]
    
    T = T0
    # Batasi agar iterasinya sebanding dengan algoritma lain (sekitar 50 iterasi)
    while len(history_fitness) <= 50:
        for _ in range(iter_per_temp):
            neighbor = random.choice(get_neighbors(current))
            neighbor_fitness = hitung_fitness(neighbor)
            delta = neighbor_fitness - current_fitness
            
            if delta > 0:
                current = neighbor
                current_fitness = neighbor_fitness
            else:
                probability = math.exp(delta / T)
                if random.random() < probability:
                    current = neighbor
                    current_fitness = neighbor_fitness
                    
            history_fitness.append(current_fitness)
            if len(history_fitness) > 50: break
            
        T = T * cooling_rate
        
    return current, current_fitness, history_fitness

# ==========================================
# 3. GENETIC ALGORITHM (GA)
# ==========================================
def tournament_selection(populasi, fitness_pop, k=3):
    # Memilih 'k' individu acak, ambil yang fitness-nya paling tinggi
    terpilih = random.sample(list(zip(populasi, fitness_pop)), k)
    terpilih.sort(key=lambda x: x[1], reverse=True)
    return terpilih[0][0]

def crossover(parent1, parent2):
    # Single-point crossover (potong di titik acak, lalu gabungkan silang)
    titik_potong = random.randint(1, JUMLAH_BARANG - 1)
    child1 = parent1[:titik_potong] + parent2[titik_potong:]
    child2 = parent2[:titik_potong] + parent1[titik_potong:]
    return child1, child2

def mutasi(solusi, mutation_rate=0.1):
    for i in range(JUMLAH_BARANG):
        if random.random() < mutation_rate:
            solusi[i] = 1 - solusi[i] # Flip bit
    return solusi

def genetic_algorithm(pop_size=20, max_gen=50, mutation_rate=0.1):
    # 1. Inisialisasi Populasi
    populasi = [generate_solusi_awal() for _ in range(pop_size)]
    history_best = []
    
    best_solusi_global = None
    best_fitness_global = 0
    
    for gen in range(max_gen + 1):
        # 2. Evaluasi Fitness
        fitness_pop = [hitung_fitness(ind) for ind in populasi]
        
        # Simpan yang terbaik di generasi ini
        best_gen_fitness = max(fitness_pop)
        best_gen_index = fitness_pop.index(best_gen_fitness)
        history_best.append(best_gen_fitness)
        
        if best_gen_fitness > best_fitness_global:
            best_fitness_global = best_gen_fitness
            best_solusi_global = populasi[best_gen_index].copy()
            
        # 3. Bentuk Generasi Baru
        populasi_baru = []
        # Elitism: Masukkan 2 solusi terbaik langsung ke generasi berikutnya
        populasi_baru.append(best_solusi_global)
        populasi_baru.append(populasi[best_gen_index])
        
        while len(populasi_baru) < pop_size:
            # Seleksi Parent
            p1 = tournament_selection(populasi, fitness_pop)
            p2 = tournament_selection(populasi, fitness_pop)
            
            # Crossover
            c1, c2 = crossover(p1, p2)
            
            # Mutasi
            c1 = mutasi(c1, mutation_rate)
            c2 = mutasi(c2, mutation_rate)
            
            populasi_baru.extend([c1, c2])
            
        # Potong jika kelebihan akibat elitism
        populasi = populasi_baru[:pop_size]
        
    return best_solusi_global, best_fitness_global, history_best

# ==========================================
# 4. TESTING & VISUALISASI KETIGA ALGORITMA
# ==========================================
if __name__ == "__main__":
    print("Mencari solusi Knapsack Problem...\n")
    MAX_ITERASI = 50
    
    # Jalankan Ketiga Algoritma
    sol_hc, fit_hc, hist_hc = steepest_ascent_hill_climbing(max_iter=MAX_ITERASI)
    print(f"Hasil Hill Climbing      : {sol_hc} | Profit: {fit_hc}")
    
    sol_sa, fit_sa, hist_sa = simulated_annealing(T0=100.0, cooling_rate=0.85, iter_per_temp=1)
    print(f"Hasil Simulated Annealing: {sol_sa} | Profit: {fit_sa}")
    
    sol_ga, fit_ga, hist_ga = genetic_algorithm(pop_size=10, max_gen=MAX_ITERASI, mutation_rate=0.1)
    print(f"Hasil Genetic Algorithm  : {sol_ga} | Profit: {fit_ga}")
    
    # --- MENAMPILKAN GRAFIK PERBANDINGAN ---
    plt.figure(figsize=(10, 6))
    
    plt.plot(hist_hc[:MAX_ITERASI+1], label='Steepest-Ascent HC', marker='o', markersize=4, color='blue', linewidth=2, alpha=0.7)
    plt.plot(hist_sa[:MAX_ITERASI+1], label='Simulated Annealing', marker='s', markersize=4, color='orange', linewidth=2, alpha=0.7)
    plt.plot(hist_ga[:MAX_ITERASI+1], label='Genetic Algorithm', marker='^', markersize=4, color='green', linewidth=2)
    
    plt.title('Perbandingan Konvergensi Algoritma Optimasi (Knapsack Problem)', fontsize=14, pad=15)
    plt.xlabel('Iterasi / Generasi', fontsize=12)
    plt.ylabel('Nilai Fitness (Total Profit Terbesar)', fontsize=12)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right', fontsize=11)
    
    plt.tight_layout()
    plt.show()