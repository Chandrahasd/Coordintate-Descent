CXXFLAGS =	-g  -Wall -fmessage-length=0

OBJS =		CD.o

LIBS =	-pthread -std=c++11

INCLUDE = -I headers/

EXT_SOURCE = headers/ConfigParser/*.cpp

EXT_OBJ = ./*.o

SOURCE = src/*.cpp 

TARGET =	CD

$(TARGET):	$(OBJS)
	$(CXX) $(INCLUDE) -o $(TARGET) $(EXT_SOURCE) $(SOURCE) $(LIBS)

all:	clean $(TARGET)

main:
	$(CXX) $(INCLUDE) -c $(SOURCE)  $(LIBS)
	$(CXX) $(INCLUDE) $(EXT_OBJ) -o $(TARGET)  $(LIBS)

ext:	
	$(CXX) $(INCLUDE) -c $(EXT_SOURCE) $(LIBS)

clean:
	rm -f $(TARGET) $(EXT_OBJ)
