<?php
class Cart {
    private $items = [];
    public function add($item) { $this->items[] = $item; return $this->total(); }
    public function total() { return count_items($this->items); }
}
function count_items($items) { return count($items); }
$cart = new Cart();
$cart->add("apple");
