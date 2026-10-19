class Engine(val power: Int) {
    fun start(): String = describe(power)
}

fun describe(p: Int): String = "power $p"

fun main() {
    val e = Engine(5)
    println(e.start())
}
