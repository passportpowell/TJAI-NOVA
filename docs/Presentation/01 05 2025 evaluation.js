import React, { useState } from 'react';
import { ChevronLeft, ChevronRight, Zap, FileCheck, AlertTriangle, Repeat, ArrowUpRight } from 'lucide-react';

const NovaPresentation = () => {
  const [currentSlide, setCurrentSlide] = useState(0);

  const slides = [
    {
      title: "Nova AI Coordinator Evaluation System Fix",
      content: (
        <div className="space-y-4">
          <p className="text-xl">Project Summary</p>
          <p>Successfully identified and resolved reliability issues in the evaluation system</p>
          <div className="mt-8 grid grid-cols-3 gap-6">
            <div className="bg-red-100 p-4 rounded-lg shadow text-center">
              <AlertTriangle className="mx-auto text-red-500 mb-2" size={28} />
              <p className="font-semibold">Problem</p>
              <p className="text-sm">Consistent 0.5/1.0 scores for all answers</p>
            </div>
            <div className="bg-yellow-100 p-4 rounded-lg shadow text-center">
              <Zap className="mx-auto text-yellow-500 mb-2" size={28} />
              <p className="font-semibold">Action</p>
              <p className="text-sm">Enhanced evaluation robustness & recovery</p>
            </div>
            <div className="bg-green-100 p-4 rounded-lg shadow text-center">
              <FileCheck className="mx-auto text-green-500 mb-2" size={28} />
              <p className="font-semibold">Result</p>
              <p className="text-sm">71.43% pass rate with varied scores</p>
            </div>
          </div>
        </div>
      )
    },
    {
      title: "The Problem",
      content: (
        <div className="space-y-4">
          <div className="bg-red-50 border-l-4 border-red-500 p-4 mb-6">
            <p className="font-semibold">Every response was evaluated with:</p>
            <ul className="list-disc pl-5 mt-2">
              <li>Score of exactly 0.5/1.0</li>
              <li>Generic feedback: "Answer provided some information"</li>
              <li>Generic weakness: "Unable to properly evaluate due to an error"</li>
              <li>0% pass rate across all answers</li>
            </ul>
          </div>
          
          <p className="mb-4">The evaluation system was consistently encountering errors and falling back to a default response instead of providing genuine assessment.</p>
          
          <div className="bg-gray-100 p-4 rounded-lg text-sm font-mono overflow-auto max-h-64">
            <p># Error fallback code in original implementation:</p>
            <p>fallback_result = {'{'}</p>
            <p className="pl-4">"score": 0.5,</p>
            <p className="pl-4">"strengths": ["Answer provided some information"],</p>
            <p className="pl-4">"weaknesses": ["Unable to properly evaluate due to an error"],</p>
            <p className="pl-4">"improvement_suggestions": ["Try providing more specific information"],</p>
            <p className="pl-4">"passed": False,</p>
            <p className="pl-4">"error": str(e)</p>
            <p>{'}'}</p>
          </div>
        </div>
      )
    },
    {
      title: "Implementation of the Fix",
      content: (
        <div className="space-y-4">
          <p className="font-semibold mb-4">Key Improvements to the Evaluation System:</p>
          
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-blue-50 p-3 rounded-lg">
              <div className="flex items-center mb-2">
                <Repeat className="text-blue-500 mr-2" size={20} />
                <p className="font-medium">Retry Mechanism</p>
              </div>
              <p className="text-sm">Added multiple attempts before falling back</p>
            </div>
            
            <div className="bg-blue-50 p-3 rounded-lg">
              <div className="flex items-center mb-2">
                <ArrowUpRight className="text-blue-500 mr-2" size={20} />
                <p className="font-medium">Fallback Model</p>
              </div>
              <p className="text-sm">Alternative LLM when primary model fails</p>
            </div>
            
            <div className="bg-blue-50 p-3 rounded-lg">
              <div className="flex items-center mb-2">
                <FileCheck className="text-blue-500 mr-2" size={20} />
                <p className="font-medium">Better JSON Handling</p>
              </div>
              <p className="text-sm">Improved extraction of partial responses</p>
            </div>
            
            <div className="bg-blue-50 p-3 rounded-lg">
              <div className="flex items-center mb-2">
                <AlertTriangle className="text-blue-500 mr-2" size={20} />
                <p className="font-medium">Enhanced Error Reporting</p>
              </div>
              <p className="text-sm">More meaningful fallback responses</p>
            </div>
          </div>
          
          <div className="mt-4 text-sm">
            <p className="font-semibold">Simplification of evaluation prompt for more reliable responses</p>
            <p>Reduced complexity and clarified requirements to increase parsing success rate</p>
          </div>
        </div>
      )
    },
    {
      title: "Results - Before vs. After",
      content: (
        <div>
          <div className="grid grid-cols-2 gap-6">
            <div className="border border-red-200 rounded-lg p-4">
              <p className="font-semibold text-center mb-4 bg-red-100 py-1 rounded">Before Fix</p>
              <div className="space-y-3 text-sm">
                <p><span className="font-medium">Evaluation Score:</span> Consistently 0.5/1.0</p>
                <p><span className="font-medium">Pass Rate:</span> 0%</p>
                <p><span className="font-medium">Feedback Quality:</span> Generic, identical</p>
                <p><span className="font-medium">Error Handling:</span> Falls to default without retries</p>
              </div>
            </div>
            
            <div className="border border-green-200 rounded-lg p-4">
              <p className="font-semibold text-center mb-4 bg-green-100 py-1 rounded">After Fix</p>
              <div className="space-y-3 text-sm">
                <p><span className="font-medium">Evaluation Score:</span> Varied (0.4-0.9/1.0)</p>
                <p><span className="font-medium">Pass Rate:</span> 71.43%</p>
                <p><span className="font-medium">Feedback Quality:</span> Specific, meaningful</p>
                <p><span className="font-medium">Error Handling:</span> Multiple recovery paths</p>
              </div>
            </div>
          </div>
          
          <div className="mt-8">
            <p className="font-medium mb-2">Sample Improved Evaluation:</p>
            <div className="bg-gray-50 p-3 rounded text-sm">
              <p><span className="font-semibold">Question:</span> Who is the king or president of that country?</p>
              <p><span className="font-semibold">Score:</span> 0.9/1.0, PASSED</p>
              <p><span className="font-semibold">Strengths:</span> Provides accurate information about the UK's monarchy, Clearly states that the UK does not have a president, Addresses the question directly</p>
              <p><span className="font-semibold">Weaknesses:</span> Does not specify context of 'that country' explicitly, Could include more details about the monarch's role</p>
            </div>
          </div>
        </div>
      )
    },
    {
      title: "Remaining Issues",
      content: (
        <div className="space-y-6">
          <div>
            <p className="font-semibold mb-2">Some Questions Still Skip Evaluation:</p>
            
            <div className="bg-yellow-50 border-l-4 border-yellow-500 p-4 mb-4">
              <p className="font-medium">Math Questions (e.g., "what 5 * 7?")</p>
              <ul className="list-disc pl-5 mt-1 text-sm">
                <li>Handled by dedicated <code>do_maths</code> function</li>
                <li>Bypasses evaluation system by design</li>
                <li>Math has deterministic answers that don't need LLM evaluation</li>
              </ul>
            </div>
            
            <div className="bg-yellow-50 border-l-4 border-yellow-500 p-4">
              <p className="font-medium">Capital Question ("What is the capital of UK?")</p>
              <ul className="list-disc pl-5 mt-1 text-sm">
                <li>System didn't properly answer the question</li>
                <li>Only echoed the question back ("What is the capital of the United Kingdom?")</li>
                <li>Suggests an upstream issue in answer generation pipeline</li>
                <li>Evaluation might be skipped when no real answer is provided</li>
              </ul>
            </div>
          </div>
          
          <div>
            <p className="font-semibold">Evaluation Success Summary:</p>
            <div className="w-full bg-gray-200 rounded-full h-4 mt-2">
              <div className="bg-green-500 h-4 rounded-full" style={{width: '71%'}}></div>
            </div>
            <div className="flex justify-between text-sm mt-1">
              <span>0%</span>
              <span className="font-medium">71.43% Passed</span>
              <span>100%</span>
            </div>
          </div>
        </div>
      )
    },
    {
      title: "Next Steps & Recommendations",
      content: (
        <div className="space-y-6">
          <div>
            <p className="font-medium mb-3">Further Improvements:</p>
            <ul className="space-y-3">
              <li className="flex items-start">
                <div className="bg-blue-100 p-1 rounded mr-2 mt-1">1</div>
                <div>
                  <p className="font-medium">Investigate the answer generation pipeline</p>
                  <p className="text-sm">Identify why the capital question wasn't properly answered</p>
                </div>
              </li>
              
              <li className="flex items-start">
                <div className="bg-blue-100 p-1 rounded mr-2 mt-1">2</div>
                <div>
                  <p className="font-medium">Enhance the monitoring system</p>
                  <p className="text-sm">Add logging of all evaluation attempts, including the ones that are skipped</p>
                </div>
              </li>
              
              <li className="flex items-start">
                <div className="bg-blue-100 p-1 rounded mr-2 mt-1">3</div>
                <div>
                  <p className="font-medium">Improve prompt engineering</p>
                  <p className="text-sm">Continue refining evaluation prompts for even more reliable results</p>
                </div>
              </li>
              
              <li className="flex items-start">
                <div className="bg-blue-100 p-1 rounded mr-2 mt-1">4</div>
                <div>
                  <p className="font-medium">Consider adding unit tests</p>
                  <p className="text-sm">Develop test cases for evaluation functions to prevent regressions</p>
                </div>
              </li>
            </ul>
          </div>
          
          <div className="bg-green-50 border border-green-200 rounded-lg p-4 mt-4">
            <p className="font-medium text-center mb-2">Overall Success</p>
            <p className="text-sm text-center">The evaluation system is now functioning properly when invoked, with a 71.43% pass rate and meaningful feedback that helps drive the fallback strategies.</p>
          </div>
        </div>
      )
    }
  ];

  const nextSlide = () => {
    setCurrentSlide((prev) => Math.min(prev + 1, slides.length - 1));
  };

  const prevSlide = () => {
    setCurrentSlide((prev) => Math.max(prev - 1, 0));
  };

  return (
    <div className="flex flex-col h-full bg-white">
      {/* Header */}
      <div className="bg-gray-800 text-white p-4">
        <h1 className="text-xl font-bold">{slides[currentSlide].title}</h1>
        <div className="text-xs mt-1 text-gray-400">Nova AI Coordinator Improvement Project</div>
      </div>
      
      {/* Content */}
      <div className="flex-grow p-6 overflow-auto">
        {slides[currentSlide].content}
      </div>
      
      {/* Footer with navigation */}
      <div className="flex justify-between bg-gray-100 p-3 border-t">
        <div className="flex space-x-1">
          <button 
            onClick={prevSlide} 
            disabled={currentSlide === 0}
            className={`p-2 rounded ${currentSlide === 0 ? 'text-gray-400' : 'text-gray-700 hover:bg-gray-200'}`}
          >
            <ChevronLeft size={20} />
          </button>
          <button 
            onClick={nextSlide} 
            disabled={currentSlide === slides.length - 1}
            className={`p-2 rounded ${currentSlide === slides.length - 1 ? 'text-gray-400' : 'text-gray-700 hover:bg-gray-200'}`}
          >
            <ChevronRight size={20} />
          </button>
        </div>
        
        <div className="text-sm text-gray-500">
          Slide {currentSlide + 1} of {slides.length}
        </div>
      </div>
    </div>
  );
};

export default NovaPresentation;