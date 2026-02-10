#include <iostream>

void crash_now() {
    int arr[5] = {1, 2, 3, 4, 5};
    // 越界写入 10000 个位置，必死无疑
    for (int i = 0; i < 10000; ++i) {
        arr[i] = 999; 
    }
}

int main() {
    crash_now();
    return 0;
}