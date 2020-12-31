CXX		= /usr/share/gcc-4.8.4/bin/g++
CXXFLAGS = -std=c++11 -DNDEBUG -mcmodel=medium -march=native -O3
RM = rm
SOURCE = src
INCLUDE = include
OBJECT = object
EXECUTABLE = benchCompression
OBJS 	= $(OBJECT)/CodecFactory.o $(OBJECT)/LinearRegressionFactory.o \
		  $(OBJECT)/varintGBTables.o $(OBJECT)/varintGUTables.o $(OBJECT)/SIMDMasks.o

$(EXECUTABLE): $(OBJS) $(OBJECT)/benchCompression.o
	$(CXX) $(CXXFLAGS) $(OBJS) $(OBJECT)/benchCompression.o -o $(EXECUTABLE)

$(OBJECT)/benchCompression.o: benchCompression.cpp $(INCLUDE)/DeltaFactory.h $(INCLUDE)/LinearRegressionFactory.h
	$(CXX) $(CXXFLAGS) -I$(INCLUDE) -c benchCompression.cpp -o $(OBJECT)/benchCompression.o


$(OBJECT)/CodecFactory.o: $(SOURCE)/CodecFactory.cpp $(INCLUDE)/CodecFactory.h
	$(CXX) $(CXXFLAGS) -I$(INCLUDE) -c $(SOURCE)/CodecFactory.cpp -o $(OBJECT)/CodecFactory.o

$(OBJECT)/LinearRegressionFactory.o: $(SOURCE)/LinearRegressionFactory.cpp $(INCLUDE)/LinearRegressionFactory.h
	$(CXX) $(CXXFLAGS) -I$(INCLUDE) -c $(SOURCE)/LinearRegressionFactory.cpp -o $(OBJECT)/LinearRegressionFactory.o


$(OBJECT)/varintGBTables.o: generate/varintGBTables.cpp
	$(CXX) $(CXXFLAGS) -I$(INCLUDE) -c generate/varintGBTables.cpp -o $(OBJECT)/varintGBTables.o

$(OBJECT)/varintGUTables.o: generate/varintGUTables.cpp
	$(CXX) $(CXXFLAGS) -I$(INCLUDE) -c generate/varintGUTables.cpp -o $(OBJECT)/varintGUTables.o

$(OBJECT)/SIMDMasks.o: generate/SIMDMasks.cpp
	$(CXX) $(CXXFLAGS) -I$(INCLUDE) -c generate/SIMDMasks.cpp -o $(OBJECT)/SIMDMasks.o

clean:
	$(RM) -f $(EXECUTABLE) 
	$(RM) -f $(OBJECT)/*
