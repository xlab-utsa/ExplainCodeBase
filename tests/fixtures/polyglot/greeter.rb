# Greets people.
class Greeter
  def initialize(name)
    @name = name
  end

  def greet
    format_name(@name)
  end

  def format_name(n)
    n.capitalize
  end
end

def main
  g = Greeter.new("bob")
  g.greet
end
