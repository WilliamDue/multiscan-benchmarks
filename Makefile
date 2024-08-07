default: bench

.PHONY: clean bench

clean:
	cd common && make clean
	cd data && make clean
	cd filter && make clean
	cd partition && make clean
	cd segreduce && make clean
	cd lexer && make clean
	
bench:
	@echo "Testing and Benching Single Pass Scan:"
	cd common && make
	@echo ""
	@echo "Testing and Benching Filter:"
	cd filter && make
	@echo ""
	@echo "Testing and Benching Partition:"
	cd partition && make
	@echo ""
	@echo "Testing and Benching Segmented Reduce:"
	cd segreduce && make
	@echo ""
	@echo "Testing and Benching Lexer:"
	cd lexer && make