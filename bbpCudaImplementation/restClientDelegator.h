#pragma once
#include <list>
#include <mutex>
#include "kernel.cuh"

class digitData;
class progressData;
struct segmentResult {
	digitData * digit;
	sJ result;
	double totalTime;
};

struct segmentRequest {
	progressData * controller;
	std::list<std::string> controlledUuids;
};

class restClientDelegator
{
private:
	std::list<segmentResult> resultsToSave;
	std::list<segmentRequest> requestsForWork;
	std::mutex resultsMtx, requestsMtx;

public:
	void addResultToQueue(digitData * digit, sJ result, double totalTime);
	void addRequestToQueue(progressData * controller, std::list<std::string> controlledUuids);
	void monitorQueues();
};

