#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <queue>
#include <chrono>
#include <atomic>
#include <omp.h>
#include <iomanip>
#include <fstream>
#include <Windows.h>

// Обновлённая версия TaskBag
class TaskPool {
private:
    std::queue<int> rowQueue;              // Очередь задач (индексы строк)
    std::mutex queueGuard;                 // Мьютекс доступа
    std::atomic<bool> doneFlag;            // Флаг завершения
    int n;                                 // Размер матрицы

    const std::vector<std::vector<int>>& srcA;
    const std::vector<std::vector<int>>& srcB;
    std::vector<std::vector<int>>& outC;

public:
    TaskPool(
        int size,
        const std::vector<std::vector<int>>& A,
        const std::vector<std::vector<int>>& B,
        std::vector<std::vector<int>>& C
    ) : n(size), srcA(A), srcB(B), outC(C), doneFlag(false)
    {
        for (int i = 0; i < size; ++i) {
            rowQueue.push(i);
        }
    }

    // Новый порядок: сначала флаг и проверка
    void mark_done() {
        doneFlag = true;
    }

    bool completed() {
        return doneFlag && rowQueue.empty();
    }

    // Получение строки из очереди
    bool fetch_row(int& rowIndex) {
        std::lock_guard<std::mutex> lock(queueGuard);
        if (rowQueue.empty()) return false;

        rowIndex = rowQueue.front();
        rowQueue.pop();
        return true;
    }

    // Выполнение вычисления строки
    void process_row(int i) {
        for (int j = 0; j < n; ++j) {
            int sum = 0;
            for (int k = 0; k < n; ++k) {
                sum += srcA[i][k] * srcB[k][j];
            }
            outC[i][j] = sum;
        }
    }
};


// Рабочий поток
void worker(TaskPool& pool) {
    int row;
    while (!pool.completed()) {
        if (pool.fetch_row(row)) {
            pool.process_row(row);
        }
        else {
            std::this_thread::yield();
        }
    }
}


// Умножение через TaskPool
std::vector<std::vector<int>> multiply_taskpool(
    int size,
    const std::vector<std::vector<int>>& A,
    const std::vector<std::vector<int>>& B,
    int threads
) {
    std::vector<std::vector<int>> C(size, std::vector<int>(size, 0));

    TaskPool pool(size, A, B, C);

    std::vector<std::thread> workers;
    workers.reserve(threads);

    for (int i = 0; i < threads; ++i) {
        workers.emplace_back(worker, std::ref(pool));
    }

    pool.mark_done();

    for (auto& t : workers) t.join();

    return C;
}


// Версия OpenMP
std::vector<std::vector<int>> multiply_openmp(
    int size,
    const std::vector<std::vector<int>>& A,
    const std::vector<std::vector<int>>& B
) {
    std::vector<std::vector<int>> C(size, std::vector<int>(size, 0));

#pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            int sum = 0;
            for (int k = 0; k < size; ++k) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
    return C;
}


// Простое умножение
std::vector<std::vector<int>> multiply_simple(
    int size,
    const std::vector<std::vector<int>>& A,
    const std::vector<std::vector<int>>& B
) {
    std::vector<std::vector<int>> C(size, std::vector<int>(size, 0));

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            int sum = 0;
            for (int k = 0; k < size; ++k) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
    return C;
}


// Генерация случайной матрицы
std::vector<std::vector<int>> generate_matrix(int size) {
    std::vector<std::vector<int>> M(size, std::vector<int>(size));
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j)
            M[i][j] = rand() % 100;

    return M;
}


// Проверка результата
bool matrices_equal(const std::vector<std::vector<int>>& X, const std::vector<std::vector<int>>& Y) {
    if (X.size() != Y.size()) return false;

    for (size_t i = 0; i < X.size(); ++i)
        for (size_t j = 0; j < X[i].size(); ++j)
            if (X[i][j] != Y[i][j]) return false;

    return true;
}


// Эксперимент
void run_benchmark() {
    std::vector<int> sizes = { 100, 200, 500, 800, 1000 };
    std::vector<int> threads = { 2, 4, 8 };
    int runs = 5;

    // --- Ровная таблица ---
    std::cout << std::left
        << std::setw(12) << "Matrix Size"
        << std::setw(10) << "Threads"
        << std::setw(18) << "TaskPool (ms)"
        << std::setw(18) << "OpenMP (ms)"
        << std::setw(18) << "Speedup TP"
        << std::setw(18) << "Speedup OMP"
        << "\n";

    std::cout << std::string(12 + 10 + 18 + 18 + 18 + 18, '-') << "\n";

    // CSV файл
    std::ofstream f("performance_data.csv");
    f << "Size;Threads;TaskPool;OpenMP;SpeedupTP;SpeedupOMP\n";

    for (int n : sizes) {
        auto A = generate_matrix(n);
        auto B = generate_matrix(n);

        auto t0 = std::chrono::high_resolution_clock::now();
        auto seq = multiply_simple(n, A, B);
        auto t1 = std::chrono::high_resolution_clock::now();
        double seq_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        for (int tnum : threads) {
            double tp_sum = 0;
            double omp_sum = 0;

            for (int r = 0; r < runs; ++r) {
                auto tp0 = std::chrono::high_resolution_clock::now();
                auto tp = multiply_taskpool(n, A, B, tnum);
                auto tp1 = std::chrono::high_resolution_clock::now();
                tp_sum += std::chrono::duration<double, std::milli>(tp1 - tp0).count();

                omp_set_num_threads(tnum);
                auto o0 = std::chrono::high_resolution_clock::now();
                auto om = multiply_openmp(n, A, B);
                auto o1 = std::chrono::high_resolution_clock::now();
                omp_sum += std::chrono::duration<double, std::milli>(o1 - o0).count();

                if (!matrices_equal(seq, tp) || !matrices_equal(seq, om))
                    std::cout << "Ошибка: результаты не совпадают!\n";
            }

            double tp_avg = tp_sum / runs;
            double omp_avg = omp_sum / runs;

            double s_tp = seq_ms / tp_avg;
            double s_omp = seq_ms / omp_avg;

            // --- Ровный вывод строки ---
            std::cout << std::left
                << std::setw(12) << n
                << std::setw(10) << tnum
                << std::setw(18) << std::fixed << std::setprecision(2) << tp_avg
                << std::setw(18) << std::fixed << std::setprecision(2) << omp_avg
                << std::setw(18) << std::fixed << std::setprecision(3) << s_tp
                << std::setw(18) << std::fixed << std::setprecision(3) << s_omp
                << "\n";

            f << n << ";" << tnum << ";" << tp_avg << ";" << omp_avg << ";" << s_tp << ";" << s_omp << "\n";
        }
    }

    f.close();
}



// MAIN
int main() {
    SetConsoleOutputCP(1251);

    std::cout << "Параллельное умножение матриц — тестирование\n\n";

    run_benchmark();
    return 0;
}
