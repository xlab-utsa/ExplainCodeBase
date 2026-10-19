#include <stdio.h>

static int twice(int x) { return 2 * x; }

int compute(int x) {
    printf("%d\n", x);
    return twice(x);
}
