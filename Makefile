# === Configuration ===
CC       := gcc
CFLAGS   := -Wall -Wextra -std=c11 -Iinclude -fopenmp
LDFLAGS  := -fopenmp

SRC_DIR   := source
INC_DIR   := include
BUILD_DIR := build

TARGET := programme.exe

# === Fichiers ===
SRCS := $(wildcard $(SRC_DIR)/*.c)
OBJS := $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(SRCS))

# === Règles ===
.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

run: all
	./$(TARGET)

clean:
	rm -rf $(BUILD_DIR) $(TARGET)