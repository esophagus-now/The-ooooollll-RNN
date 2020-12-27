main: *.cpp *.h
	clang++ -o main -std=c++17 -g -Wall *.cpp

test: tests/*.cpp *.cpp *.h
	clang++ -std=c++17 -o test -g -Wall tests/tensor_test.cpp $(ls *.cpp | grep -v "main.cpp")

clean:
	rm -rf main 
	rm -rf test