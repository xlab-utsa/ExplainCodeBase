package demo;

/** A shape with an area. */
abstract class Shape {
    Shape() {}
    abstract double area();
    String describe() { return "area=" + area(); }
}

class Square extends Shape {
    private final double side;

    Square(double side) {
        super();
        this.side = side;
    }

    double area() { return Util.square(side); }
}

class Util {
    static double square(double x) { return x * x; }

    static String run() {
        Square s = new Square(2);
        return s.describe();
    }
}
