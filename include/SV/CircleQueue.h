#pragma once

#include <iostream>
#include <vector>
#include <mutex>
#include <condition_variable>

typedef struct FrameData {
    std::vector<unsigned char> color_data;
    std::vector<uint16_t> depth_data;
};

class CircleQueue {
public:
    CircleQueue(int cap = 20)
            : _cap(cap) {
        _front = 0;
        _rear = 0;
        buffer.resize(_cap);
    }

    ~CircleQueue() {
        buffer.clear();
    }

    bool isQueueFull() {
        return (_rear + 1) % _cap == _front;
    }

    bool isQueueEmpty() {
        return _front == _rear;
    }

    int getQueueSize() {
        return (_rear + _cap - _front) % _cap;
    }

    void Put(const FrameData &frame) {
        std::unique_lock<std::mutex> lck(mtx);
        if (isQueueFull()) {
            _nullCond.wait(lck);
        }

        buffer[_rear] = frame;
        _rear = (_rear + 1) % _cap;

        _dataCond.notify_all();
    }

    void Get(FrameData &frame) {
        std::unique_lock<std::mutex> lck(mtx);
        if (isQueueEmpty()) {
            _dataCond.wait(lck);
        }

        frame = buffer[_front];
        _front = (_front + 1) % _cap;

        _nullCond.notify_all();
    }

    void noMoveGet(FrameData &frame) {
        std::unique_lock<std::mutex> lck(mtx);
        if (isQueueEmpty()) {
            _dataCond.wait(lck);
        }

        frame = buffer[_front];

        _nullCond.notify_all();
    }

    std::mutex mtx;
    std::condition_variable _dataCond;
    std::condition_variable _nullCond;

    int _front;
    int _rear;

    int _cap;
    std::vector<FrameData> buffer;
};