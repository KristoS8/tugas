import re
import math
from decimal import Decimal

import numpy as np
import unittest
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import utils

# group9 = []
# member_names = ["kristo", "Dzaki", "Ferdiansyah", "Marselina", "Nuryanti", "zalfa", "adelia"]

# for name in member_names:
#     pattern = r"o"
#     if re.search(pattern, name):
#         group9.append(name.upper())
#         print(group9)
#     else:
#         group9.append(name)
    
# print(group9)

one_dimensional_array = np.array([10,12,14,5,13,13,15])
print(one_dimensional_array)

# for i in one_dimensional_array:
#     print(i)

b = np.arange(10, dtype=int)
print(b)

c = np.arange(0,100,10.3, dtype=float)
print(c)

d = np.linspace(0,100,5, dtype=int) # default type for linespace is float
print(d)

print(d.dtype)

k = np.array(["welcome to hello world!"])
print(k)
print(k.dtype) # 23-character (23) unicode string (U) on a little-endian architecture (<)

m = np.ones(3)
print(m)

n = np.zeros(5)
print(n)

empty = np.empty(5)
print(empty)

random = np.random.rand(5)
print(random)

two_dim_array = np.array([[1,2,3],[4,5,6]])
print(two_dim_array)
print(two_dim_array[1][1])


print("=======================================")
one_dim_array = np.array([1,2,3,4,5,6,7,8,9,10])
multi_dim_array = np.reshape(one_dim_array, (5, 2)) # reshape mengubah dimensi array dari 1 dimensi k=menjadi 2 dimensi dengan 5 baris dan 2 kolom, alternatif cara dalam membuat array 2 dimensi menggunakan array 1 dimensi

print(multi_dim_array)
print(multi_dim_array.ndim)
print(multi_dim_array.shape)
print(multi_dim_array.size) #Returns total number of elements

array1 = np.array([1,2,3])
array2 = np.array([10,20,30], dtype=float)

adding = array1 + array2
print(adding)

subtracting = array2 - array1
print(subtracting)

multiplication = array1 * array2
print(multiplication)

print(array2*2.33)

# indexing
print(array1[2])

# slicing

# If no value is passed to start, it is assumed start = 0, if no value is passed for end, it is assumed that end = length of array and if no value is passed to step, it is assumed step = 1.
slicearray = one_dim_array[2:8:2]
slicearray2 = one_dim_array[4:]
slicearray3 = one_dim_array[:5]
slicearray4 = one_dim_array[::-1]
slicearray5 = one_dim_array[::]
print(slicearray)
print(slicearray2)
print(slicearray3)
print("slice 4: ", slicearray4)
print(slicearray5)

# stack
a1 = np.array([[1,1], 
               [2,2]])
a2 = np.array([[3,3],
              [4,4]])
print(f'a1:\n{a1}')
print(f'a2:\n{a2}')

ver_stack = np.vstack((a1, a2))
print(f"vertikal stack: \n{ver_stack}")
hori_stack = np.hstack((a1, a2))
print(f"horizontal stack: \n{hori_stack}")

# Representing dan sloving sistem persamaan linear menggunakan matriks
# non singular matriks
A = np.array([[4,-3,1], [2,1,3], [-1,2,-5]], dtype=np.dtype(float))
B = np.array([-10,0,17], dtype=np.dtype(float))

print("Matriks A:")
print(A)
print("Matriks B:")
print(B)

print(f"shape of A: {np.shape(A)}")
print(f"shape of B: {np.shape(B)}")

x = np.linalg.solve(A, B)
print(x)

d = np.linalg.det(A)
print(f"determinan of Matriks A: {d:.2f}")

# singular matriks

A_2= np.array([
        [1, 1, 1],
        [0, 1, -3],
        [2, 1, 5]
    ], dtype=np.dtype(float))

b_2 = np.array([2, 1, 0], dtype=np.dtype(float))

# print(np.linalg.solve(A_2, b_2))

d_2 = np.linalg.det(A_2)

print(f"Determinant of matrix A_2: {d_2:.2f}")

# Algoritma Gaussian (Eliminasi)
# swap rows
def swap_rows(M, row_index_1, row_index_2):
    M = M.copy()
    M[[row_index_1 , row_index_2]] = M[[row_index_2, row_index_1]]
    return M


M = np.array([
    [1,2,3],
    [-6,-2,5],
    [9,-2,8]
])

print(M)
print("=================")

M_swapped = swap_rows(M, 0 , 2)
print(M_swapped)

#

print("\n")

# fungsi untuk mencari nilai bukan 0 pertama di dalam column array
def get_index_first_non_zero_value_from_column (M, column, starting_row): # parameternya itu arraynya, kolom berapa dan baris berapa
    column_array = M[starting_row:,column] # slicing dari array index ke baris berapa dan kolom berapa (ngambil kolom) koma buat memisahkan indeks baris dan kolom.
    print(column_array)
    for i, val in enumerate(column_array): # i mulai dari 0, enumerate buat mengulang suatu list sekaligus mendapatkan index posisi dalam array
        # print("val: ",val)
        if not np.isclose(val, 0, atol=1e-5): # mengecek jika not true maka pernyataan dalam if tidak dijalankan, jika not false maka true dan dijalankan, is close untuk mengecek apakah nilai val itu mendekati nilai 0 dengan toleransi 1e-5, yaitu 0.000001
            index = i + starting_row # tambahin i dan row saat ini
            return index
    return -1 # mengembalikan -1 jika nilai columnya 0 semua


N = np.array([
    [0, 5, -3 ,6 ,8],
    [0, 6, 3, 8, 1],
    [0, 0, 0, 0, 0],
    [0, 0, 0 ,0 ,7],
    [0, 2, 1, 0, 4]
])

# print(N)

print(get_index_first_non_zero_value_from_column(N, column=1, starting_row=2))

print("\n")
# fungsi untuk mencari nilai bukan 0 pertama dalam sebuah baris

def get_index_first_non_zero_value_from_row (M, row, augmented=False):

    M = M.copy()

    if augmented == True: # augmented untuk menghilangkan kolom terakhir dalam baris
        M = M[:,:-1]

    row_array = M[row]
    for i, val in enumerate(row_array):
        if not np.isclose(val, 0, atol=1e-5):
            return i

    return -1


k = np.array([
    [0, 5, -3 ,6 ,8],
    [0, 6, 3, 8, 1],
    [0, 0, 0, 0, 0],
    [0, 0, 0 ,5 ,7],
    [0, 2, 1, 0, 4]
])

print(k)

print(get_index_first_non_zero_value_from_row(k, 2))
print(get_index_first_non_zero_value_from_row(k, 0))
print(get_index_first_non_zero_value_from_row(k, 3, augmented=True))

# stack/ menggabungkan 2 matriks

print("\n")

def augmented_matriks (A, B):

    augmented_M = np.hstack((A,B))
    return augmented_M


A = np.array([[1,2,3], [3,4,5], [4,5,6]])
B = np.array([[1], [5], [7]])

print(augmented_matriks(B, A))

print("\n")

# Linear transformation dan neural network

def T(v):
    w = np.zeros((3,1))
    w[0,0] = 3*v[0,0]
    w[2,0] = -2*v[1,0]
    return w


kristo = np.array([[3],[5]])
w = T(kristo)

print(f"Original vector: \n{kristo} \n\n Transformation: \n{w}")


def S(v):
    w = np.zeros((3,1))
    w[0,0] = 4*v[0,0]
    w[1,0] = -1*v[1,0]
    w[2,0] = 5*v[2,0]

    return w

nilai = np.array([[2],[3],[4]])

print(f"original: \n{nilai}, \n\nhasil transformasi: \n{S(nilai)}")


u = np.array([[1], [-2]])
v = np.array([[2], [4]])

k = 7

print("T(k*v):\n", T(k*v), "\n k*T(v):\n", k*T(v), "\n\n")
print("T(u+v):\n", T(u+v), "\n\n T(u)+T(v):\n", T(u)+T(v))

print("\n")
print("\n")

# Transformations Defined as a Matrix Multiplication
def M(v):
    w = np.array([[3,0],[0,0],[0,-2]])
    print(f"Transformation Matriks: \n {w}")

    a = w @ v
    return a

arr = np.array([[8], [5]])
print("Matriks ori: \n ", arr)
print("Hasil transformasi linear: \n" , M(arr))