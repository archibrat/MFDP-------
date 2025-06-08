class TestMLIntegration:
    """Интеграционные тесты ML-компонентов"""
    
    @pytest.mark.asyncio
    async def test_full_prediction_flow(self, test_patient_data):
        """Тест полного потока предсказания"""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            # Создание предсказания
            response = await ac.post("/api/predictions/predict", json={
                "patient_data": test_patient_data,
                "prediction_types": ["no_show_risk"],
                "include_explanation": True
            })
            
            assert response.status_code == 200
            prediction_result = response.json()
            prediction_id = prediction_result["prediction_id"]
            
            # Получение истории
            patient_id = test_patient_data["client_id"]
            history_response = await ac.get(f"/api/predictions/history/{patient_id}")
            
            assert history_response.status_code == 200
            history = history_response.json()
            
            assert len(history) > 0
            assert any(p["id"] == prediction_id for p in history)
    
    @pytest.mark.asyncio
    async def test_model_performance_monitoring(self):
        """Тест мониторинга производительности модели"""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            # Получение аналитики
            response = await ac.get("/api/predictions/analytics/performance?days=7")
            
            assert response.status_code == 200
            analytics = response.json()
            
            assert "accuracy" in analytics
            assert "f1_score" in analytics
            assert "dates" in analytics
            
            if analytics["accuracy"]:
                assert all(0 <= acc <= 1 for acc in analytics["accuracy"])
    
    @pytest.mark.asyncio
    async def test_concurrent_predictions(self, test_patient_data):
        """Тест конкурентных предсказаний"""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            # Создание множественных запросов
            tasks = []
            for i in range(10):
                patient_data = test_patient_data.copy()
                patient_data["client_id"] = f"TEST_{i:03d}"
                patient_data["booking_id"] = f"BOOK_{i:03d}"
                
                task = ac.post("/api/predictions/predict", json={
                    "patient_data": patient_data,
                    "prediction_types": ["no_show_risk"]
                })
                tasks.append(task)
            
            # Выполнение всех запросов
            responses = await asyncio.gather(*tasks)
            
            # Проверка результатов
            assert all(r.status_code == 200 for r in responses)
            results = [r.json() for r in responses]
            
            # Проверка уникальности prediction_id
            prediction_ids = [r["prediction_id"] for r in results]
            assert len(set(prediction_ids)) == len(prediction_ids)

# Конфигурация для pytest
if __name__ == "__main__":
    pytest.main([__file__, "-v"])