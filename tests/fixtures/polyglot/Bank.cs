namespace Demo {
    class Account {
        public Account(int balance) { Balance = balance; }
        public int Balance { get; set; }
        public void Deposit(int amount) { Balance = Add(Balance, amount); }
        static int Add(int a, int b) { return a + b; }
    }

    class Savings : Account {
        public Savings() : base(0) {}
        public void Grow() { this.Deposit(1); }
    }
}
