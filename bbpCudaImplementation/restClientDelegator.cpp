#include "restClientDelegator.h"

#include <boost/beast/core.hpp>
#include <boost/beast/version.hpp>
#include <boost/asio/ssl/error.hpp>
#include <boost/asio/connect.hpp>
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
#include "digitData.h"
#include "kernel.cuh"
#include "progressData.h"

std::string apiKey = "";
std::string domain = "";
std::string targetPort = "";

namespace ip = boost::asio::ip;
namespace http = boost::beast::http;
namespace ssl = boost::asio::ssl;

//modified from boost examples: https://www.boost.org/doc/libs/1_69_0/libs/beast/example/http/client/async/http_client_async.cpp
// Performs an HTTP GET and prints the response
class session : public std::enable_shared_from_this<session>
{
	ip::tcp::resolver::results_type resolved;
	ssl::stream<ip::tcp::socket> stream_;
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
	std::string apiKey;

public:
	// Resolver and socket require an io_context
	explicit
		session(boost::asio::io_context& ioc, ssl::context& sslCtx, ip::tcp::resolver::results_type resolved,
			char const* host,
			char const* target,
			std::string apiKey,
			int version)
		: resolved(resolved)
		, stream_(ioc, sslCtx)
		, timeout(ioc)
		, host(host)
		, port(port)
		, target(target)
		, version(version)
		, apiKey(apiKey)
	{
	}

	// Start the asynchronous operation
	void
		run(std::function<void(const boost::property_tree::ptree&)> processResponse, std::function<void()> failHandler, std::string body, http::verb verb)
	{
		if (!SSL_set_tlsext_host_name(stream_.native_handle(), host))
		{
			boost::system::error_code ec{ static_cast<int>(::ERR_get_error()), boost::asio::error::get_ssl_category() };
			std::cerr << ec.message() << std::endl;
			return;
		}

		this->processResponse = processResponse;
		this->failHandler = failHandler;
		// Set up an HTTP GET request message
		req_.version(version);
		req_.method(verb);
		req_.target(target);
		req_.set(http::field::host, host);
		req_.set(http::field::user_agent, BOOST_BEAST_VERSION_STRING);
		req_.set("apiKey", apiKey);
		req_.body() = body;
		req_.set(http::field::content_type, "application/json");
		req_.set(http::field::content_length, body.size());

		//cancel the request and close the socket on timeout
		timeout.expires_from_now(boost::posix_time::seconds(2));
		timeout.async_wait([&](boost::system::error_code const &ec) {
			if (ec == boost::asio::error::operation_aborted) return;
			stream_.async_shutdown(
				std::bind(
					&session::on_shutdown,
					shared_from_this(),
					std::placeholders::_1));
		});

		// Make the connection on the IP address we get from a lookup
		boost::asio::async_connect(
			stream_.next_layer(),
			resolved.begin(),
			resolved.end(),
			std::bind(
				&session::on_connect,
				shared_from_this(),
				std::placeholders::_1));
	}

	void
		on_connect(boost::system::error_code ec)
	{
		if (ec) {
			std::cerr << "Connection error: " << ec.message() << std::endl;
			return failHandler();
		}

		// Perform the SSL handshake
		stream_.async_handshake(
			ssl::stream_base::client,
			std::bind(
				&session::on_handshake,
				shared_from_this(),
				std::placeholders::_1));
	}

	void
		on_handshake(boost::system::error_code ec)
	{
		if (ec) {
			std::cerr << "Handshake error: " << ec.message() << std::endl;
			return failHandler();
		}

		// Send the HTTP request to the remote host
		http::async_write(stream_, req_,
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

		if (ec) {
			std::cerr << ec.message() << std::endl;
			return;
		}

		// Receive the HTTP response
		http::async_read(stream_, buffer_, res_,
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

		if (ec) {
			std::cerr << ec.message() << std::endl;
			return failHandler();
		}

		if (res_.result_int() != 200) {
			std::cerr << "Error response code: " << res_.result_int() << std::endl;
			return failHandler();
		}

		boost::property_tree::ptree pt;

		std::stringstream ss(res_.body());
		boost::property_tree::read_json(ss, pt);

		processResponse(pt);

		// Gracefully close the socket
		stream_.async_shutdown(
			std::bind(
				&session::on_shutdown,
				shared_from_this(),
				std::placeholders::_1));

		// If we get here then the connection is closed gracefully
	}

	void
		on_shutdown(boost::system::error_code ec)
	{
		if (ec == boost::asio::error::eof)
		{
			// Rationale:
			// http://stackoverflow.com/questions/25587403/boost-asio-ssl-async-shutdown-always-finishes-with-an-error
			ec.assign(0, ec.category());
		}
		if (ec)
			return;

		// If we get here then the connection is closed gracefully
	}

};

std::string hexConvert(uint64 value) {
	std::stringstream s;
	s << std::hex << std::setfill('0') << std::setw(16) << value;
	return s.str();
}

std::string hexConvert(double value) {
	std::stringstream s;
	s << std::hexfloat << std::setprecision(13) << value;
	return s.str();
}

void restClientDelegator::retryOnFail(apiCall * toRetry) {
	queueMtx.lock();
	toRetry->timeValid = std::chrono::steady_clock::now() + std::chrono::seconds(2);
	apiCallQueue.push(toRetry);
	queueMtx.unlock();
}

void restClientDelegator::noopFail(apiCall * failed) {
	delete failed;
}

void restClientDelegator::quitUponSegmentExpirationSuccess(apiCall * succeeded, progressData * controller, uint64 remoteId, const boost::property_tree::ptree pt) {
	if (!pt.empty()) {
		bool reservationExtended = pt.get<bool>("success");
		if (!reservationExtended) {
			controller->setStopCheck(remoteId);
		}
	}
	delete succeeded;
}

void restClientDelegator::noopSuccess(apiCall * succeeded, const boost::property_tree::ptree pt) {
	delete succeeded;
}

void restClientDelegator::processWorkGetResponse(progressData * data, restClientDelegator * returnToSender, apiCall * call, const boost::property_tree::ptree& pt) {
	if (!pt.empty()) {
		uint64 sumEnd = pt.get<uint64>("segmentEnd");
		uint64 segmentBegin = pt.get<uint64>("segmentStart");
		uint64 exponent = pt.get<uint64>("exponent");
		uint64 remoteId = pt.get<uint64>("id");
		digitData * workUnit = new digitData(sumEnd, exponent, segmentBegin, remoteId);
		data->assignWork(workUnit);
		delete call;
	}
	else {
		call->failHandle();
	}
}

void restClientDelegator::addResultPutToQueue(digitData * workSegment, uint128 result, double totalTime) {
	std::stringstream body, endpoint;
	boost::property_tree::ptree pt;
	pt.put("most-significant-word", hexConvert(result.msw));
	pt.put("least-significant-word", hexConvert(result.lsw));
	pt.put("time", hexConvert(totalTime));
	boost::property_tree::json_parser::write_json(body, pt);
	endpoint << "/pushSegment/" << workSegment->remoteId;
	apiCall * call = new apiCall();
	call->successHandle = std::bind(&restClientDelegator::noopSuccess, call, std::placeholders::_1);
	call->body = body.str();
	call->endpoint = endpoint.str();
	call->verb = http::verb::put;
	call->timeValid = std::chrono::steady_clock::now();
	call->failHandle = std::bind(&restClientDelegator::retryOnFail, this, call);
	queueMtx.lock();
	apiCallQueue.push(call);
	queueMtx.unlock();
}

void restClientDelegator::addWorkGetToQueue(progressData * controller) {
	apiCall * call = new apiCall();
	call->endpoint = "/getSegment/" + controller->controlledUuids();
	call->body = "";
	call->verb = http::verb::get;
	call->failHandle = std::bind(&restClientDelegator::retryOnFail, this, call);
	call->timeValid = std::chrono::steady_clock::now();
	call->successHandle = std::bind(&restClientDelegator::processWorkGetResponse, controller, this, call, std::placeholders::_1);
	queueMtx.lock();
	apiCallQueue.push(call);
	queueMtx.unlock();
}

void restClientDelegator::addReservationExtensionPutToQueue(digitData * workSegment, double progress, double timeElapsed, progressData * controller) {
	std::stringstream endpoint;
	endpoint << "/extendSegmentReservation/" << workSegment->remoteId;
	apiCall * call = new apiCall();
	call->successHandle = std::bind(&restClientDelegator::quitUponSegmentExpirationSuccess, call, controller, workSegment->remoteId, std::placeholders::_1);
	call->body = "";
	call->endpoint = endpoint.str();
	call->verb = http::verb::put;
	call->timeValid = std::chrono::steady_clock::now();
	call->failHandle = std::bind(&restClientDelegator::noopFail, call);
	queueMtx.lock();
	apiCallQueue.push(call);
	queueMtx.unlock();
}

void restClientDelegator::addProgressUpdatePutToQueue(digitData * workSegment, uint128 intermediateResult, uint64 computedUpTo, double timeElapsed) {
	std::stringstream body, endpoint;
	boost::property_tree::ptree pt;
	pt.put("most-significant-word", hexConvert(intermediateResult.msw));
	pt.put("least-significant-word", hexConvert(intermediateResult.lsw));
	pt.put("continue-from", hexConvert(computedUpTo));
	pt.put("time", hexConvert(timeElapsed));
	boost::property_tree::json_parser::write_json(body, pt);
	endpoint << "/progressUpdate/" << workSegment->remoteId;
	apiCall * call = new apiCall();
	call->successHandle = std::bind(&restClientDelegator::noopSuccess, call, std::placeholders::_1);
	call->body = body.str();
	call->endpoint = endpoint.str();
	call->verb = http::verb::put;
	call->timeValid = std::chrono::steady_clock::now();
	call->failHandle = std::bind(&restClientDelegator::noopFail, call);
	queueMtx.lock();
	apiCallQueue.push(call);
	queueMtx.unlock();
}

void restClientDelegator::processQueue(boost::asio::io_context& ioc, ssl::context& sslCtx, const std::chrono::steady_clock::time_point validBefore) {
	queueMtx.lock();
	while (!apiCallQueue.empty() && apiCallQueue.top()->timeValid < validBefore) {
		const apiCall * call = apiCallQueue.top();
		std::make_shared<session>(ioc, sslCtx, resolvedResults, domain.c_str(), call->endpoint.c_str(), apiKey, 11)->run(call->successHandle, call->failHandle, call->body, call->verb);
		apiCallQueue.pop();
	}
	queueMtx.unlock();
}

bool restClientDelegator::resolve(boost::asio::io_context& ioc, std::string host, std::string port) {
	if (nextResolve < std::chrono::steady_clock::now()) {
		ip::tcp::resolver resolver(ioc);
		boost::system::error_code ec;
		resolvedResults = resolver.resolve(host.c_str(), port.c_str(), ec);
		if (!ec) {
			nextResolve = std::chrono::steady_clock::now() + std::chrono::minutes(5);
			lastResolveSuccessful = true;
		}
		else {
			std::cerr << ec.message() << std::endl;
			nextResolve = std::chrono::steady_clock::now() + std::chrono::seconds(1);
			lastResolveSuccessful = false;
		}
	}
	return lastResolveSuccessful;
}

void restClientDelegator::monitorQueues() {
	boost::asio::io_context ioc;
	// The SSL context is required, and holds certificates
	ssl::context ctx{ ssl::context::sslv23_client };
	ctx.load_verify_file("rootcert.txt");
	// Verify the remote server's certificate
	ctx.set_verify_mode(ssl::verify_peer);

	nextResolve = std::chrono::steady_clock::now() - std::chrono::seconds(1);
	lastResolveSuccessful = false;
	while (!globalStopSignal) {
		if (resolve(ioc, domain, targetPort)) {
			std::chrono::steady_clock::time_point validBefore = std::chrono::steady_clock::now();
			processQueue(ioc, ctx, validBefore);
		}
		ioc.poll();//process any handlers currently ready on the context (using this instead of ::run avoids getting stuck waiting on a timeout to expire for a dead request)
		std::this_thread::sleep_for(std::chrono::milliseconds(2));//rest between checking the queues for work
	}
}