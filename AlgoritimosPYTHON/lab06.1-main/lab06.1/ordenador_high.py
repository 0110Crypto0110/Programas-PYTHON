class Sorter:
    def __init__(self):
        pass

    def particiona(self, arr, low, high):
        pivot = arr[high]
        index = low 
    
        for j in range(low, high):
            if arr[j] < pivot:
                arr[index], arr[j] = arr[j], arr[index]
                index += 1
    
        arr[index], arr[high] = arr[high], arr[index]
        return index

    def quick_sort(self, arr, low, high) -> [int]:
        if low < high:
            pi = particiona(arr, low, high)
            quick_sort(arr, low, pi - 1)
            quick_sort(arr, pi + 1, high)

    def merge(self, left, right) -> [int]:
        result = []
        i = j = 0

        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1

        result.extend(left[i:])
        result.extend(right[j:])
        return result

    def merge_sort(self, arr) -> [int]:
        if len(arr) <= 1:
            return arr

        mid = len(arr) // 2
        left_half = arr[:mid]
        right_half = arr[mid:]

        left_half = merge_sort(left_half)
        right_half = merge_sort(right_half)

        return merge(left_half, right_half)
        
