SRC_DIR   := source
INC_DIR   := include
BUILD_DIR := build

# === Configuration ===
CC       := gcc
CFLAGS   := -Wall -Wextra -std=c11 -I$(INC_DIR) -fopenmp -O3 -MMD -MP
LDFLAGS  := -fopenmp -lm

TARGET := programme.exe

# === Fichiers ===
SRCS := $(wildcard $(SRC_DIR)/*.c $(SRC_DIR)/*/*.c)
OBJS := $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(SRCS))

# === Règles ===
.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

run: all
	./$(TARGET)

clean:
	rm -rf $(BUILD_DIR) $(TARGET)

#-include $(OBJS:.o=.d)