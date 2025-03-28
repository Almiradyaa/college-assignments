#include <stdio.h>

// Fungsi Binary Search (Iteratif)
int binary_search(int a[], int n, int size) {
    int l = 0, r = size - 1;

    while (l <= r) {
        int mid = l + (r - l) / 2;

        if (a[mid] == n)
            return mid;
        else if (a[mid] > n)
            r = mid - 1;
        else
            l = mid + 1;
    }

    return -1;  // Jika elemen tidak ditemukan
}

int main() {
    // Array dengan angka kelipatan tambah 2 dari 2 sampai 32
    int arr[] = {2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32};
    int size = sizeof(arr) / sizeof(arr[0]);
    int target;

    printf("=== PROGRAM BINARY SEARCH ===\n");
    printf("Masukkan angka yang ingin dicari (Masukkan -1 untuk keluar)\n");

    while (1) {
        printf("\nMasukkan angka: ");
        scanf("%d", &target);

        if (target == -1) {
            printf("Terima kasih telah menggunakan program ini!\n");
            break;
        }

        int result = binary_search(arr, target, size);

        if (result != -1)
            printf("Angka ditemukan di indeks ke-%d\n", result);
        else
            printf("Angka tidak ditemukan dalam daftar.\n");

    }

    return 0;
}
