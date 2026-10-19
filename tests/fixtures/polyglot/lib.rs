pub struct Counter { n: u32 }

impl Counter {
    pub fn new() -> Self { Counter { n: 0 } }
    pub fn incr(&mut self) -> u32 { self.n = step(self.n); self.n }
}

fn step(n: u32) -> u32 { n + 1 }

pub fn run() -> u32 {
    let mut c = Counter::new();
    c.incr()
}
