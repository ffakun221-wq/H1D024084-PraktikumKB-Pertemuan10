import random
import matplotlib.pyplot as plt

from inisiasipopulasi import inisialisasi_populasi
from EvaluasiFitness import hitung_fitness

from selection import roulette_wheel_selection, tournament_selection
from crossover import one_point_crossover, two_point_crossover, uniform_crossover
from mutation import swap_mutation, inversion_mutation, uniform_mutation

barang = [
    ("Barang1", 10, 5),
    ("Barang2", 40, 4),
    ("Barang3", 30, 6),
    ("Barang4", 50, 3),
    ("Barang5", 35, 7)
]

def run_ga(jumlah_generasi, jumlah_populasi, prob_crossover, prob_mutasi, kapasitas_tas):
    jumlah_gen = len(barang)
    
    populasi = inisialisasi_populasi(jumlah_populasi, jumlah_gen)
    
    best_fitness_list = []
    worst_fitness_list = []
    avg_fitness_list = []
    all_fitness = []
    
    best_individu = None
    best_fitness_overall = 0
    
    for generasi in range(jumlah_generasi):
        fitness_populasi = [hitung_fitness(individu, barang, kapasitas_tas) for individu in populasi]
        
        best_fitness = max(fitness_populasi)
        worst_fitness = min(fitness_populasi)
        avg_fitness = sum(fitness_populasi) / len(fitness_populasi)
        
        best_fitness_list.append(best_fitness)
        worst_fitness_list.append(worst_fitness)
        avg_fitness_list.append(avg_fitness)
        all_fitness.append(fitness_populasi.copy())
        
        if best_fitness > best_fitness_overall:
            best_fitness_overall = best_fitness
            index_best = fitness_populasi.index(best_fitness)
            best_individu = populasi[index_best]
            
        new_populasi = []
        used_indices = []
        
        while len(new_populasi) < jumlah_populasi:
          
            # 1. SELEKSI (Berdasarkan Digit 8: Roulette Wheel Selection)
            
            parent1, idx1 = roulette_wheel_selection(populasi, fitness_populasi)
            used_indices.append(idx1)
            
            available_indices = [i for i in range(len(populasi)) if i not in used_indices]
            if not available_indices:
                used_indices = [idx1]
                available_indices = [i for i in range(len(populasi)) if i != idx1]
                
            parent2, _ = roulette_wheel_selection(
                [populasi[i] for i in available_indices],
                [fitness_populasi[i] for i in available_indices]
            )
            used_indices.append(available_indices[_])
            
            
            # 2. CROSSOVER (Berdasarkan Digit 4: Two-Point Crossover)
            
            if random.random() < prob_crossover:
                anak1, anak2 = two_point_crossover(parent1, parent2)
            else:
                anak1, anak2 = parent1[:], parent2[:]
                
           
            # 3. MUTASI (Berdasarkan Penjumlahan 8+4=12 -> Akhiran 2: Uniform)
           
            if random.random() < prob_mutasi:
                anak1 = uniform_mutation(anak1)
            if random.random() < prob_mutasi:
                anak2 = uniform_mutation(anak2)
                
            new_populasi.extend([anak1, anak2])
            
        populasi = new_populasi[:jumlah_populasi]
        
    plt.figure(figsize=(12, 7))
    
    for i in range(jumlah_generasi):
        x = [i+1]*len(all_fitness[i])
        y = all_fitness[i]
        plt.scatter(x, y, color='gray', alpha=0.1)
        
    plt.plot(range(1, jumlah_generasi+1), best_fitness_list, color='blue', label='Fitness Tertinggi')
    plt.plot(range(1, jumlah_generasi+1), worst_fitness_list, color='yellow', label='Fitness Terendah')
    plt.plot(range(1, jumlah_generasi+1), avg_fitness_list, color='red', label='Fitness Rata-rata')
    
    plt.title('Perkembangan Nilai Fitness - Praktikum 10 (NIM: 84)')
    plt.xlabel('Generasi')
    plt.ylabel('Nilai Fitness')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    selected_items = [barang[i][0] for i in range(len(best_individu)) if best_individu[i] == 1]
    selected_value = hitung_fitness(best_individu, barang, kapasitas_tas)
    selected_weight = sum([barang[i][2] for i in range(len(best_individu)) if best_individu[i] == 1])
    
    print(f"Nilai Keuntungan (Fitness) Terbaik: {selected_value}")
    print(f"Total Ukuran Gudang Terpakai: {selected_weight} / {kapasitas_tas}")
    print("Barang Terpilih:")
    for item in selected_items:
        print(f"- {item}")

if __name__ == "__main__":
    run_ga(
        jumlah_generasi=50,
        jumlah_populasi=20,
        prob_crossover=0.5,
        prob_mutasi=0.1,
        kapasitas_tas=15
    )