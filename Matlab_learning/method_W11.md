# sub2ind()
convert subscripts to linear indices

1st col | 2nd col | 3rd col
-------|--------|--------
(1, 1) | (1, 2) | (1, 3)
(2, 1) | (2, 2) | (2, 3)
(3, 1) | (3, 2) | (3, 3)
###
1st col | 2nd col | 3rd col
---|---|---
 1 | 4 | 7 
 2 | 5 | 8 
 3 | 6 | 9 
###
linear indices<br />
1, 2, 3,  4,  5,  6,  7,  8,  9

# One-Hot enciding
> Example Output: D(k),
> - D is the matrix of 1-by-10
> - k is the number represent 1 to 9

 k | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 0
 --------|---|---|---|---|---|---|---|---|---|---
 1st row | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0
 2nd row | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0
 3rd row | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0
 4th row | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0
 5th row | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0
 6th row | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0
 7th row | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0
 8th row | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0
 9th row | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0
10th row | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1
