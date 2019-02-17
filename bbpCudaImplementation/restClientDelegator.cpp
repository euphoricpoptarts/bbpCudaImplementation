#include "restClientDelegator.h"

//modified from boost examples: https://www.boost.org/doc/libs/1_69_0/libs/beast/example/http/client/async/http_client_async.cpp

#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/version.hpp>
#include <boost/asio/connect.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/asio/deadline_timer.hpp>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <chrono>
#include <sstream>
#include <queue>
#include <list>
#include "digitData.hpp"
#include "kernel.cuh"
#include "progressData.h"

namespace ip = boost::asio::ip;       // from <boost/asio/ip/tcp.hpp>
namespace http = boost::beast::http;    // from <boost/beast/http.hpp>

										//------------------------------------------------------------------------------

										// Report a failure
void
fail(boost::system::error_code ec, char const* what)
{
	std::cerr << what << ": " << ec.message() << "\n";
}

// Performs an HTTP GET and prints the response
class session : public std::enable_shared_from_this<session>
{
	ip::tcp::resolver resolver_;
	ip::tcp::socket socket_;
	boost::asio::deadline_timer timeout;
	boost::beast::flat_buffer buffer_; // (Must persist between reads)
	http::request<http::string_body> req_;
	http::response<http::string_body> res_;
	char const* host;
	char const* port;
	char const* target;
	int version;
	std::function<void(const boost::property_tree::ptree&)> processResponse;
	std::function<void()> failHandler;

public:
	// Resolver and socket require an io_context
	explicit
		session(boost::asio::io_context& ioc, char const* host,
			char const* port,
			char const* target,
			int version)
		: resolver_(ioc)
		, socket_(ioc)
		, timeout(ioc)
		, host(host)
		, port(port)
		, target(target)
		, version(version)
	{
	}

	// Start the asynchronous operation
	void
		run(std::function<void(const boost::property_tree::ptree&)> processResponse, std::function<void()> failHandler, std::string body, http::verb verb)
	{
		this->processResponse = processResponse;
		this->failHandler = failHandler;
		// Set up an HTTP GET request message
		req_.version(version);
		req_.method(verb);
		req_.target(target);
		req_.set(http::field::host, host);
		req_.set(http::field::user_agent, BOOST_BEAST_VERSION_STRING);
		req_.body() = body;
		req_.set(http::field::content_type, "application/json");
		req_.set(http::field::content_length, body.size());
		// Look up the domain name
		resolver_.async_resolve(
			host,
			port,
			std::bind(
				&session::on_resolve,
				shared_from_this(),
				std::placeholders::_1,
				std::placeholders::_2));
	}

	void
		on_resolve(
			boost::system::error_code ec,
			ip::tcp::resolver::results_type results)
	{
		if (ec)
			return fail(ec, "resolve");

		// Make the connection on the IP address we get from a lookup
		boost::asio::async_connect(
			socket_,
			results.begin(),
			results.end(),
			std::bind(
				&session::on_connect,
				shared_from_this(),
				std::placeholders::_1));
	}

	void
		on_connect(boost::system::error_code ec)
	{
		if (ec)
			return fail(ec, "connect");

		// Send the HTTP request to the remote host
		http::async_write(socket_, req_,
			std::bind(
				&session::on_write,
				shared_from_this(),
				std::placeholders::_1,
				std::placeholders::_2));
	}

	void
		on_write(
			boost::system::error_code ec,
			std::size_t bytes_transferred)
	{
		boost::ignore_unused(bytes_transferred);

		if (ec)
			return fail(ec, "write");
		
		//cancel the request and close the socket on timeout
		timeout.expires_from_now(boost::posix_time::seconds(2));
		timeout.async_wait([&](boost::system::error_code const &ec) {
			if (ec == boost::asio::error::operation_aborted) return;
			socket_.cancel();
		});

		// Receive the HTTP response
		http::async_read(socket_, buffer_, res_,
			std::bind(
				&session::on_read,
				shared_from_this(),
				std::placeholders::_1,
				std::placeholders::_2));
	}

	void
		on_read(
			boost::system::error_code ec,
			std::size_t bytes_transferred)
	{
		timeout.cancel();
		boost::ignore_unused(bytes_transferred);

		// not_connected happens sometimes so don't bother reporting it.
		if (ec && ec != boost::system::errc::not_connected) {
			std::cout << ec.message() << std::endl;
			return failHandler();
		}

		boost::property_tree::ptree pt;

		std::stringstream ss(res_.body());
		boost::property_tree::read_json(ss, pt);

		processResponse(pt);

		// Gracefully close the socket
		socket_.shutdown(ip::tcp::socket::shutdown_both, ec);

		// If we get here then the connection is closed gracefully
	}

	static void processRequest(progressData * data, std::list<std::string> controlledUuids, restClientDelegator * returnToSender, std::function<void()> noResultHandler, const boost::property_tree::ptree& pt) {
		if (!pt.empty()) {
			uint64 sumEnd = std::stoull(pt.get<std::string>("segmentEnd"));
			uint64 segmentBegin = std::stoull(pt.get<std::string>("segmentStart"));
			uint64 exponent = std::stoull(pt.get<std::string>("exponent"));
			digitData * workUnit = new digitData(sumEnd, exponent, segmentBegin, 0);
			data->assignWork(workUnit);
		}
		else {
			noResultHandler();
		}
	}

	static void processResult(const boost::property_tree::ptree& pt) {
		/*if (!pt.empty()) {
			for (auto iter = pt.begin(); iter != pt.end(); iter++) {
				std::cout << iter->first << ":" << iter->second.get_value<std::string>() << std::endl;
			}
			std::cout << "Getting by value .get('segment'): " << pt.get<std::string>("segment") << std::endl;
		}
		else {
			std::cout << pt.get_value<std::string>() << std::endl;
		}*/
	}

	static void noopProcess(const boost::property_tree::ptree& pt) {}

	static void requestFail(progressData * data, std::list<std::string> controlledUuids, restClientDelegator * returnToSender) {
		returnToSender->addRequestToQueue(data, controlledUuids, std::chrono::high_resolution_clock::now() + std::chrono::seconds(2));
	}

	static void resultFail(digitData * digit, sJ result, double totalTime, restClientDelegator * returnToSender) {
		std::cout << "Somethings fucked" << std::endl;
		returnToSender->addResultToQueue(digit, result, totalTime, std::chrono::high_resolution_clock::now() + std::chrono::seconds(2));
	}

	static void noopFail() {}
};

void restClientDelegator::addResultToQueue(digitData * digit, sJ result, double totalTime) {
	resultsMtx.lock();
	resultsToSave.push(segmentResult{ digit, result, totalTime, std::chrono::high_resolution_clock::now() });
	resultsMtx.unlock();
}

void restClientDelegator::addRequestToQueue(progressData * controller, std::list<std::string> controlledUuids) {
	requestsMtx.lock();
	requestsForWork.push(segmentRequest{ controller, controlledUuids, std::chrono::high_resolution_clock::now() });
	requestsMtx.unlock();
}

void restClientDelegator::addResultToQueue(digitData * digit, sJ result, double totalTime, std::chrono::high_resolution_clock::time_point validAfter) {
	resultsMtx.lock();
	resultsToSave.push(segmentResult{ digit, result, totalTime, validAfter });
	resultsMtx.unlock();
}

void restClientDelegator::addRequestToQueue(progressData * controller, std::list<std::string> controlledUuids, std::chrono::high_resolution_clock::time_point validAfter) {
	requestsMtx.lock();
	requestsForWork.push(segmentRequest{ controller, controlledUuids, validAfter });
	requestsMtx.unlock();
}

void restClientDelegator::addProgressUpdateToQueue(digitData * digit, double progress, double timeElapsed) {
	progressMtx.lock();
	progressUpdates.push(progressUpdate{digit, progress, timeElapsed, std::chrono::high_resolution_clock::now() });
	progressMtx.unlock();
}

void restClientDelegator::processResultsQueue(boost::asio::io_context& ioc, std::chrono::high_resolution_clock::time_point validBefore) {
	resultsMtx.lock();
	while (!resultsToSave.empty() && resultsToSave.top().timeValid < validBefore) {
		const segmentResult& result = resultsToSave.top();
		std::function<void(boost::property_tree::ptree)> successF = std::bind(&session::processResult, std::placeholders::_1);
		std::function<void()> failF = std::bind(&session::resultFail, result.digit, result.result, result.totalTime, this);
		std::stringstream body, endpoint;
		body << "{ \"most-significant-word\": \"" << std::hex << std::setfill('0') << std::setw(16) << result.result.s[1];
		body << "\", \"least-significant-word\": \"" << std::setw(16) << result.result.s[0];
		body << "\", \"time\": \"" << std::hexfloat << std::setprecision(13) << result.totalTime << "\"}";
		endpoint << "/pushSegment/" << result.digit->segmentBegin << "/" << result.digit->sumEnd << "/" << result.digit->startingExponent;
		delete result.digit;
		std::make_shared<session>(ioc, "127.0.0.1", "5000", endpoint.str().c_str(), 11)->run(successF, failF, body.str(), http::verb::put);
		resultsToSave.pop();
	}
	resultsMtx.unlock();
}

void restClientDelegator::processRequestsQueue(boost::asio::io_context& ioc, std::chrono::high_resolution_clock::time_point validBefore) {
	requestsMtx.lock();
	while (!requestsForWork.empty() && requestsForWork.top().timeValid < validBefore) {
		const segmentRequest& request = requestsForWork.top();
		std::function<void()> failF = std::bind(&session::requestFail, request.controller, request.controlledUuids, this);
		std::function<void(boost::property_tree::ptree)> successF = std::bind(&session::processRequest, request.controller, request.controlledUuids, this, failF, std::placeholders::_1);
		std::make_shared<session>(ioc, "127.0.0.1", "5000", "/getSegment", 11)->run(successF, failF, "", http::verb::get);
		requestsForWork.pop();
	}
	requestsMtx.unlock();
}

void restClientDelegator::processProgressUpdatesQueue(boost::asio::io_context& ioc, std::chrono::high_resolution_clock::time_point validBefore) {
	progressMtx.lock();
	while (!progressUpdates.empty() && progressUpdates.top().timeValid < validBefore) {
		const progressUpdate& progress = progressUpdates.top();
		std::function<void()> failF = std::bind(&session::noopFail);
		std::function<void(boost::property_tree::ptree)> successF = std::bind(&session::noopProcess, std::placeholders::_1);
		std::stringstream endpoint;
		endpoint << "/extendSegmentReservation/" << progress.digit->segmentBegin << "/" << progress.digit->sumEnd << "/" << progress.digit->startingExponent;
		std::make_shared<session>(ioc, "127.0.0.1", "5000", endpoint.str().c_str(), 11)->run(successF, failF, "", http::verb::get);
		progressUpdates.pop();
	}
	progressMtx.unlock();
}

void restClientDelegator::monitorQueues() {
	boost::asio::io_context ioc;
	while (!stop) {
		std::chrono::high_resolution_clock::time_point validBefore = std::chrono::high_resolution_clock::now();
		processResultsQueue(ioc, validBefore);
		processRequestsQueue(ioc, validBefore);
		processProgressUpdatesQueue(ioc, validBefore);
		ioc.poll();//process any handlers currently ready on the context (using this instead of ::run avoids getting stuck waiting on a timeout to expire for a dead request)
		std::this_thread::sleep_for(std::chrono::milliseconds(2));//rest between checking the queues for work
	}
}