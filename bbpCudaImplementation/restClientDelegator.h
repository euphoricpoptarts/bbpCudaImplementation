#pragma once
#define _WIN32_WINNT 0x0A00//defines as windows 10, must be defined even for 64-bit builds
#include <boost/heap/priority_queue.hpp>
#include <boost/asio/io_context.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/beast/http.hpp>
#include <boost/asio/ssl/stream.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <mutex>
#include <list>
#include "kernel.cuh"
#include <chrono>

extern std::string apiKey;
extern std::string domain;
extern std::string targetPort;

class digitData;
class progressData;

struct apiCall {
	std::string endpoint, body;
	boost::beast::http::verb verb;
	std::function<void()> failHandle;
	std::function<void(boost::property_tree::ptree)> successHandle;
	std::chrono::steady_clock::time_point timeValid;

	bool operator()(const apiCall* a, const apiCall* b) const {
		return a->timeValid > b->timeValid;
	}
};

class restClientDelegator
{
private:
	boost::heap::priority_queue<apiCall*, boost::heap::compare<apiCall>> apiCallQueue;
	std::mutex queueMtx;
	boost::asio::ip::tcp::resolver::results_type resolvedResults;
	std::chrono::steady_clock::time_point nextResolve;
	bool lastResolveSuccessful;
	bool resolve(boost::asio::io_context& ioc, std::string host, std::string port);
	void processQueue(boost::asio::io_context& ioc, boost::asio::ssl::context& sslCtx, const std::chrono::steady_clock::time_point validBefore, bool resolveStatus);
	void retryOnFail(apiCall * toRetry);
	static void noopFail(apiCall * failed);
	static void quitUponSegmentExpirationSuccess(apiCall * succeeded, progressData * controller, uint64 remoteId, const boost::property_tree::ptree pt);
	static void noopSuccess(apiCall * succeeded, const boost::property_tree::ptree pt);
	static void processWorkGetResponse(progressData * data, restClientDelegator * returnToSender, apiCall * call, const boost::property_tree::ptree& pt);

public:
	void addResultPutToQueue(digitData * digit, uint128 result, double totalTime);
	void addWorkGetToQueue(progressData * controller);
	void addReservationExtensionPutToQueue(digitData * digit, double progress, double timeElapsed, progressData * controller);
	void addProgressUpdatePutToQueue(digitData * digit, uint128 intermediateResult, uint64 computedUpTo, double timeElapsed);
	void monitorQueues();
};
