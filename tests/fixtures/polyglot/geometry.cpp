#include <cmath>

class Point {
 public:
  Point(double x, double y) : x_(x), y_(y) {}
  double norm() const;
 private:
  double x_, y_;
};

double Point::norm() const { return std::sqrt(sq(x_) + sq(y_)); }

static double sq(double v) { return v * v; }

int main() {
  Point p(3, 4);
  return static_cast<int>(p.norm());
}
