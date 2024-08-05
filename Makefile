default: bench

.PHONY: clean

clean:
	cd common && make clean
	cd data && make clean
	cd filter && make clean
	cd partition && make clean
	cd segreduce && make clean
	cd lexer && make clean
	
