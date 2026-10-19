package main

import "fmt"

// Server serves things.
type Server struct{ port int }

// NewServer builds a server.
func NewServer(port int) *Server {
	return &Server{port: port}
}

func (s *Server) Start() {
	s.listen()
	fmt.Println("started")
}

func (s *Server) listen() {}

func main() {
	srv := NewServer(8080)
	srv.Start()
}
