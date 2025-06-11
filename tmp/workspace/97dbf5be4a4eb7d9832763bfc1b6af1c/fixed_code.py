:
The first line of input contains integer n, both 0 < n ≤ 10^10.
The second line of the input contains string s, of length n, and consists of digits from 0 to 9.

Output:
Single line containing the number of beautiful partitions.

Example:
Input:
4
"10352"
   
Output:
9

Explanation:
There are three beautiful partitions: ["1", "2", "3"], ["103", "3", "2"], ["1035", "2", "3"].

Edit: I have to learn more about string and numbers, but I am looking at them as number of characters since getting just the ASCII values would give a big integer result, and none of the strings can be long enough.

 
```python
def count_solutions(s):
    s = s.replace("=", "")
    diff = ""
    buckets = {}
    
    for c in s:
        if c != "=":
            diff += str(ord(c))
            
    buckets[diff] = ("1", "2", "3")

    # TODO include more rules
    
    return buckets.values()[0]


class Solution:
    def countBeautifulPartitions(self, A):
        """
        :