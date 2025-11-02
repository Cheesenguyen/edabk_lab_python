#include <stdio.h>
#include <string.h> 
#include <math.h>

// Khai báo hằng số cho dữ liệu
#define NUM_SAMPLES 5       // số lượng mẫu dữ liệu 
#define NUM_FEATURES 2      // số lượng features

// khai báo hàm

/**
 * @brief Dự đoán giá trị y từ MỘT mẫu (sample) x_features (mảng 1D).
 * Đây là công thức: y = (w[0]*x[0] + w[1]*x[1] + ...) + b
 */
double predict(double w[], double b, double x_sample_features[], int num_features);

/**
 * @brief Tính toán lỗi (Cost) trung bình (MSE) trên TOÀN BỘ tập dữ liệu.
 */
double calculate_cost(double w[], double b, int n_samples, int n_features, 
                      double x[][n_features], double y[]);

/**
 * @brief Tính toán gradient (đạo hàm) cho mảng w[] và cho b.
 * @param[out] grad_w_out Mảng để lưu trữ gradient của từng w[j].
 * @param[out] grad_b_out Con trỏ để lưu trữ gradient của b.
 */
void calculate_gradients(double w[], double b, int n_samples, int n_features, 
                         double x[][n_features], double y[], 
                         double grad_w_out[], double* grad_b_out);

/**
 * @brief Cập nhật giá trị mảng w[] và b dựa trên gradient.
 */
void update_parameters(double w[], double* b, double grad_w[], double grad_b, 
                       double learning_rate, int num_features);

/**
 * @brief Chạy vòng lặp huấn luyện chính.
 */
void train(double w[], double* b, int n_samples, int n_features, 
           double x[][n_features], double y[], 
           double learning_rate, int num_steps);



int main() {
    // 1. Khởi tạo dữ liệu
    // Ví dụ: Dự đoán giá nhà (y) dựa trên Feature 0: Diện tích (m2) Feature 1: Số phòng ngủ
    
    // Dữ liệu x mảng 2D: x[NUM_SAMPLES][NUM_FEATURES]
    double x[NUM_SAMPLES][NUM_FEATURES] = {
        {100.0, 3.0}, // Mẫu 0: 100m2, 3 phòng
        {150.0, 4.0}, // Mẫu 1: 150m2, 4 phòng
        { 80.0, 2.0}, // Mẫu 2: 80m2, 2 phòng
        {120.0, 3.0}, // Mẫu 3: 120m2, 3 phòng
        {200.0, 5.0}  // Mẫu 4: 200m2, 5 phòng
    };
    
    // y 1D: giá nhà
    double y[NUM_SAMPLES] = {3.0, 4.5, 2.5, 3.5, 5.5};

    // 2. Khởi tạo tham số
    double learning_rate = 0.01; 
    int num_steps = 10000;          // số lần lặp
    
    double w[NUM_FEATURES] = {0.0, 0.0};    // mảng với 2 phần tử , tương ứng 2 features
    double b = 0.0; 

    // Mảng để lưu giá trị thống kê cho việc chuẩn hóa
    double mean[NUM_FEATURES];     // Mảng lưu giá trị trung bình
    double std_dev[NUM_FEATURES];  // Mảng lưu độ lệch chuẩn

    //3. Chuẩn hóa dữ liệu
    for (int j = 0; j < NUM_FEATURES; j++) {
        // 1. Tính Mean cho feature j 
        double sum = 0.0;
        for (int i = 0; i < NUM_SAMPLES; i++) {
            sum += x[i][j];
        }
        mean[j] = sum / NUM_SAMPLES;

        // 2. Tính Standard Deviation (Độ lệch chuẩn) cho feature j
        double variance_sum = 0.0;
        for (int i = 0; i < NUM_SAMPLES; i++) {
            variance_sum += (x[i][j] - mean[j]) * (x[i][j] - mean[j]);
        }
        std_dev[j] = sqrt(variance_sum / NUM_SAMPLES);

        // Tránh lỗi chia cho 0 nếu tất cả giá trị trong 1 feature giống hệt nhau
        if (fabs(std_dev[j]) < 1e-9) { 
            std_dev[j] = 1.0;
        }

        printf("Feature %d: Mean = %f, StdDev = %f\n", j, mean[j], std_dev[j]);
    }
    // tạo mảng dữ liệu đã chuẩn hóa
    double x_normalized[NUM_SAMPLES][NUM_FEATURES];
    for (int i = 0; i < NUM_SAMPLES; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            x_normalized[i][j] = (x[i][j] - mean[j]) / std_dev[j];
        }
    }

    //4. Training
    train(w, &b, NUM_SAMPLES, NUM_FEATURES, x_normalized, y, learning_rate, num_steps);        // chạy vòng lặp huấn luyện
    
    printf("Kết quả tìm được:\n");
    printf("w[0] (diện tích) = %.4f\n", w[0]);
    printf("w[1] (số phòng)  = %.4f\n", w[1]);
    printf("b (bias)         = %.4f\n", b);
    printf("\n");

    //5. Inference
    // Dữ liệu gốc 
    double nha_moi_1_raw[NUM_FEATURES] = {130.0, 3.0}; // 130m2, 3 phòng
    double nha_moi_2_raw[NUM_FEATURES] = {180.0, 4.0}; // 180m2, 4 phòng

    // Chuẩn hóa dữ liệu mới đã chuẩn hóa
    double nha_moi_1_norm[NUM_FEATURES];
    double nha_moi_2_norm[NUM_FEATURES];
    
    for (int j = 0; j < NUM_FEATURES; j++) {
        nha_moi_1_norm[j] = (nha_moi_1_raw[j] - mean[j]) / std_dev[j];
        nha_moi_2_norm[j] = (nha_moi_2_raw[j] - mean[j]) / std_dev[j];
    }
    // Dự đoán bằng dữ liệu ĐÃ CHUẨN HÓA
    double gia_du_doan_1 = predict(w, b, nha_moi_1_norm, NUM_FEATURES);
    printf("Dự đoán cho nhà (130m2, 3 phòng): %.4f tỷ\n", gia_du_doan_1);

    double gia_du_doan_2 = predict(w, b, nha_moi_2_norm, NUM_FEATURES);
    printf("Dự đoán cho nhà (180m2, 4 phòng): %.4f tỷ\n", gia_du_doan_2);

    return 0;
}


/**
 * @brief Chạy vòng lặp huấn luyện chính.
 */
void train(double w[], double* b, int n_samples, int n_features, 
           double x[][n_features], double y[], 
           double learning_rate, int num_steps) {
    
    // grad_w BÂY GIỜ LÀ MỘT MẢNG
    double grad_w[n_features];
    double grad_b;
    
    for (int i = 0; i < num_steps; i++) {
        // 1. Tính gradient (hướng dốc)
        calculate_gradients(w, *b, n_samples, n_features, x, y, grad_w, &grad_b);

        // 2. Cập nhật tham số (bước xuống dốc)
        update_parameters(w, b, grad_w, grad_b, learning_rate, n_features);

        }
}

/**
 * @brief Dự đoán giá trị y từ MỘT mẫu (sample) x_features (mảng 1D).
 * Đây là công thức: y = (w[0]*x[0] + w[1]*x[1] + ...) + b
 */
double predict(double w[], double b, double x_sample_features[], int num_features) {
    double y_pred = 0.0;        // giá nhà dự đoán
    
    // Tính tích vô hướng (dot product) của w và x
    for (int j = 0; j < num_features; j++) {
        y_pred += w[j] * x_sample_features[j];
    }
    
    // Cộng bias b
    y_pred += b;
    
    return y_pred;
}

/**
 * @brief Tính toán gradient (đạo hàm) cho mảng w[] và cho b.
 */
void calculate_gradients(double w[], double b, int n_samples, int n_features, 
                         double x[][n_features], double y[], 
                         double grad_w_out[], double* grad_b_out) {
    
    // Khởi tạo các biến tổng gradient về 0
    double grad_w_sum[n_features];
    // Khởi tạo mảng về 0 
    memset(grad_w_sum, 0, n_features * sizeof(double)); 
    
    double grad_b_sum = 0.0;

    // Vòng lặp qua TẤT CẢ CÁC MẪU (samples)
    for (int i = 0; i < n_samples; i++) {
        // x[i] là mảng 1D chứa các features của mẫu thứ i
        double y_pred = predict(w, b, x[i], n_features); 
        double error = y[i] - y_pred;

        // Cập nhật gradient cho TỪNG w[j]
        for (int j = 0; j < n_features; j++) {
            // Đạo hàm của w[j] bị ảnh hưởng bởi x[i][j]
            // grad_w_sum[j] += -x[i][j] * error;
            grad_w_sum[j] += -x[i][j] * error;
        }
        
        // Gradient của b không đổi
        grad_b_sum += -error;
    }

    // Tính gradient trung bình
    for (int j = 0; j < n_features; j++) {
        grad_w_out[j] = (2.0 / n_samples) * grad_w_sum[j];
    }
    *grad_b_out = (2.0 / n_samples) * grad_b_sum;
}

/**
 * @brief Cập nhật giá trị mảng w[] và b dựa trên gradient.
 */
void update_parameters(double w[], double* b, double grad_w[], double grad_b, 
                       double learning_rate, int num_features) {
    
    // Cập nhật bias (như cũ)
    *b = *b - learning_rate * grad_b;
    
    // Cập nhật TỪNG w[j] trong mảng w
    for (int j = 0; j < num_features; j++) {
        w[j] = w[j] - learning_rate * grad_w[j];
    }
}

/**
 * @brief Tính toán lỗi (Cost) trung bình (MSE) trên TOÀN BỘ tập dữ liệu.
 */
/*
double calculate_cost(double w[], double b, int n_samples, int n_features, 
                      double x[][n_features], double y[]) {
    
    double total_cost = 0.0;
    
    // Lặp qua tất cả các mẫu
    for (int i = 0; i < n_samples; i++) {
        double y_pred = predict(w, b, x[i], n_features); // Dùng hàm predict
        double error = y[i] - y_pred;
        total_cost += error * error; // Cộng dồn bình phương lỗi
    }
    return total_cost / n_samples; // Trả về lỗi trung bình
}
*/