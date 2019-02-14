#pragma once
#include <boost/heap/priority_queue.hpp>
#include <mutex>
#include <list>
#include "kernel.cuh"
#include <chrono>

class digitData;
class progressData;
struct segmentResult {
	digitData * digit;
	sJ result;
	double totalTime;
	std::chrono::high_resolution_clock::time_point timeValid;
};

struct segmentRequest {
	progressData * controller;
	std::list<std::string> controlledUuids;
	std::chrono::high_resolution_clock::time_point timeValid;
};

struct comparatorX {
	bool operator()(const segmentResult& a, const segmentResult& b) const
	{
		return a.timeValid < b.timeValid;
	}
	bool operator()(const segmentRequest& a, const segmentRequest& b) const
	{
		return a.timeValid < b.timeValid;
	}
};

class restClientDelegator
{
private:
	boost::heap::priority_queue<segmentResult, boost::heap::compare<comparatorX>> resultsToSave;
	boost::heap::priority_queue<segmentRequest, boost::heap::compare<comparatorX>> requestsForWork;
	std::mutex resultsMtx, requestsMtx;

public:
	void addResultToQueue(digitData * digit, sJ result, double totalTime);
	void addRequestToQueue(progressData * controller, std::list<std::string> controlledUuids);
	void addResultToQueue(digitData * digit, sJ result, double totalTime, std::chrono::high_resolution_clock::time_point validAfter);
	void addRequestToQueue(progressData * controller, std::list<std::string> controlledUuids, std::chrono::high_resolution_clock::time_point validAfter);
	void monitorQueues();
};

