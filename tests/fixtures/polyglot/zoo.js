// Keeps track of animals.
class Zoo {
  constructor(name) {
    this.name = name;
    this.animals = [];
  }

  add(animal) {
    this.animals.push(animal);
    return count(this.animals);
  }
}

function count(list) {
  return list.length;
}

const makeZoo = (name) => {
  const zoo = new Zoo(name);
  zoo.add("lion");
  return zoo;
};

module.exports = { makeZoo };
