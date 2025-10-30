#include <stdio.h>

/**
 * @brief Tính toán lỗi (Cost) trung bình (MSE) tại một thời điểm.
 */
double calculate_cost(double w, double b, int N, double x[], double y[]);

/**
 * @brief Tính toán gradient (đạo hàm) cho w và b.
 * @param[out] grad_w_out Con trỏ để lưu trữ gradient của w.
 * @param[out] grad_b_out Con trỏ để lưu trữ gradient của b.
 */
void calculate_gradients(double w, double b, int N, double x[], double y[], 
                         double* grad_w_out, double* grad_b_out);

/**
 * @brief Cập nhật giá trị w và b dựa trên gradient.
 */
void update_parameters(double* w, double* b, double grad_w, double grad_b, 
                       double learning_rate);

/**
 * @brief Chạy vòng lặp huấn luyện chính.
 */
void train(double* w, double* b, int N, double x[], double y[], double learning_rate, int num_steps);

/**
 * @brief Dự đoán giá trị y mới từ một giá trị x,
 * sử dụng w và b đã huấn luyện.
 */
double predict(double w, double b, double x_new);


int main() {
    // Khởi tạo dữ liệu và tham số
    double x[] = {9.0, 24.0, 3.0, 4.0, 23.7};     // Đầu vào
    double y[] = {5.1, 1.9, -30.0, 4.1, 6.9};     // Đầu ra thực tế
    int N = 5;                                  // Số lượng điểm dữ liệu

    double learning_rate = 0.001; // learning rate
    int num_steps = 1000;       // Số lần lặp
    
    // Tham số mô hình (bắt đầu từ 0)
    double w = 0.0;
    double b = 0.0;

    // TRAINING: tìm w và b
    train(&w, &b, N, x, y, learning_rate, num_steps);       // tìm giá trị của w và b
    printf("Kết quả (w, b) tối ưu tìm được cho phương trình: y = %.4f * x + %.4f\n", w, b);
    printf("\n");
    
    // INFERENCE: dự đoạn y dựa trên x cho trước
    double x_new_1 = 6.0;
    double y_pred_1 = predict(w, b, x_new_1);
    printf("Dự đoán cho x1 = %.1f là y1 = %.4f\n", x_new_1, y_pred_1);
    
    double x_new_2 = 7.0;
    double y_pred_2 = predict(w, b, x_new_2);
    printf("Dự đoán cho x2 = %.1f là y2 = %.4f\n", x_new_2, y_pred_2);

    return 0;
}

/**
 * @brief Chạy vòng lặp huấn luyện chính.
 */
void train(double* w, double* b, int N, double x[], double y[], 
           double learning_rate, int num_steps) {
    
    for (int i = 0; i < num_steps; i++) {
        double grad_w, grad_b;

        // 1. Tính gradient (hướng dốc)
        calculate_gradients(*w, *b, N, x, y, &grad_w, &grad_b);

        // 2. Cập nhật tham số
        update_parameters(w, b, grad_w, grad_b, learning_rate);
    }
}

/**
 * @brief Tính toán gradient (đạo hàm) cho w và b.
 */
void calculate_gradients(double w, double b, int N, double x[], double y[], 
                         double* grad_w_out, double* grad_b_out) {
    
    double grad_w_sum = 0.0;
    double grad_b_sum = 0.0;

    for (int j = 0; j < N; j++) {
        double y_pred = predict(w, b, x[j]);    // lấy giá trị dự đoán của y
        double error = y[j] - y_pred;           // tính sai số (khoảng cách giữa giá trị dự đoán và thật)

        grad_w_sum += -x[j] * error;          // tính tổng   
        grad_b_sum += -error;
    }
    // tính gradient trung bình
    *grad_w_out = (1.0 / N) * grad_w_sum;
    *grad_b_out = (1.0 / N) * grad_b_sum;
}

/**
 * @brief Cập nhật giá trị w và b dựa trên gradient.
 */
void update_parameters(double* w, double* b, double grad_w, double grad_b, 
                       double learning_rate) {
    
    // Thay đổi giá trị tại địa chỉ mà con trỏ trỏ tới
    *w = *w - learning_rate * grad_w;
    *b = *b - learning_rate * grad_b;
}

/**
 * @brief Tính toán lỗi (Cost) trung bình (MSE) tại một thời điểm.
 */
double calculate_cost(double w, double b, int N, double x[], double y[]) {
    double total_cost = 0.0;
    for (int j = 0; j < N; j++) {
        double y_pred = predict(w, b, x[j]); // Dùng hàm predict
        double error = y[j] - y_pred;
        total_cost += error * error; // Cộng dồn bình phương lỗi
    }
    return total_cost / N; // Trả về lỗi trung bình
}

/**
 * @brief Dự đoán giá trị y mới từ một giá trị x.
 */
double predict(double w, double b, double x_new) {
    return w * x_new + b;       // tính toán giá trị y mới từ x
}