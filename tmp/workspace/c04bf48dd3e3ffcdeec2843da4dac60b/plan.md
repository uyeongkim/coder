, a_2, ..., a_n (1 ≤ a_i ≤ n), the values of the cards.

Output

The maximum possible score represented as a sum of elements in the consecutive segment chosen by Alice.

Example:

```
Input:  1
2
3

Output: 8
Explanation: Alice chooses segment [1, 2] to make the maximum score.
```

Solution

```python
from typing import List
from collections import deque

def minimum_score(cards: List[int]) -> int:
    string = ''.join(str(c) for c in cards)
    cur = 0
    seen = set()
    for i in range(len(string)):
        if string[i] not in seen:
            seen.add(string[i])
            cur += int(string[i])
            cur -= int(string[i - 1])
# `cur` is not going to be because of integer manipulations: it will overflow
    total = cur
    for i in range(1, len(string)-1):
        cur -= int(string[i])
        cur += int(string[i - 1])
        total += cur    
    return total
    
if __name__ == '__main__':
    try:
        n = int(input())
        cards = list(map(int